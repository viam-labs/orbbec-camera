from io import BytesIO
import struct
import sys
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from google.protobuf.timestamp_pb2 import Timestamp
from typing_extensions import Self
from viam.components.camera import Camera
from viam.errors import NotSupportedError
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.media.utils.pil import pil_to_viam_image
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName, ResponseMetadata
from viam.proto.component.camera import (
    GetPropertiesResponse,
    IntrinsicParameters,
    DistortionParameters,
)
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

try:
    import pyorbbecsdk
    from pyorbbecsdk import (
        AlignFilter,
        Pipeline,
        Config,
        Context,
        OBFormat,
        OBSensorType,
        OBStreamType,
        PointCloudFilter,
    )

    print(f"pyorbbecsdk version: {pyorbbecsdk.get_version()}")
except ImportError:
    print("Error: pyorbbecsdk is not installed or cannot be imported.")
    print("Please check build dependencies.")
    sys.exit(1)


class Orbbec(Camera, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "camera"), "orbbec")

    MIN_DEPTH = 20  # 20mm
    MAX_DEPTH = 10000  # 10000mm

    pipeline: Optional[Pipeline] = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Camera component.
        The default implementation sets the name from the `config` parameter and then calls `reconfigure`.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both implicit and explicit)

        Returns:
            Self: The resource
        """
        return super().new(config, dependencies)

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any implicit dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Sequence[str]: A list of implicit dependencies
        """
        attrs = struct_to_dict(config.attributes)
        sensor_list = attrs.get("sensors", ["color", "depth"])

        if len(sensor_list) == 0:
            raise Exception("'sensors' array must not be empty.")

        return []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both implicit and explicit)
        """
        attrs = struct_to_dict(config.attributes)
        self.sensor_list: List[str] = list(attrs.get("sensors", ["color", "depth"]))

        context = Context()
        device_list = context.query_devices()

        device_count = device_list.get_count()
        if device_count == 0:
            raise Exception(
                "No Orbbec devices found. Please connect a device and try again."
            )

        device = device_list.get_device_by_index(0)
        device_info = device.get_device_info()
        self.logger.debug(
            f"Device name: {device_info.get_name()}, PID: {device_info.get_pid()}, Serial Number: {device_info.get_serial_number()}"
        )
        if self.pipeline is not None:
            self.pipeline.stop
            self.pipeline = None

        self.pipeline = Pipeline(device)
        self.config = Config()

        assert self.pipeline is not None

        if "color" in self.sensor_list:
            color_profile_list = self.pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
            color_profile = color_profile_list.get_default_video_stream_profile()
            self.config.enable_stream(color_profile)

        if "depth" in self.sensor_list:
            depth_profile_list = self.pipeline.get_stream_profile_list(
                OBSensorType.DEPTH_SENSOR
            )
            depth_profile = depth_profile_list.get_default_video_stream_profile()
            self.config.enable_stream(depth_profile)
            self.pipeline.enable_frame_sync()

        self.pipeline.start(self.config)

    async def get_image(
        self,
        mime_type: str = "",
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> ViamImage:
        if self.pipeline is None:
            self.logger.error("No device connected.")
            return

        self.logger.debug(f"get_image - mime_type: {mime_type}")
        if not mime_type:
            mime_type = CameraMimeType.JPEG

        main_sensor = self.sensor_list[0]

        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return
        frame = None

        if main_sensor == "depth":
            frame = frames.get_depth_frame()

        if main_sensor == "color":
            frame = frames.get_color_frame()

        if frame is None:
            return

        return self.process_frame_as_image(frame, mime_type, sensor=main_sensor)

    def process_frame_as_image(
        self, frame, mime_type, raw=False, sensor="color"
    ) -> ViamImage:
        width = frame.get_width()
        height = frame.get_height()

        if sensor == "depth":
            if mime_type == CameraMimeType.VIAM_RAW_DEPTH:
                # Get the raw data as uint16 (2 bytes per pixel)
                depth_data = np.frombuffer(frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((height, width))

                # Encode into Viam's raw depth format
                # Required format: Magic number(8 bytes) + width(8 bytes) + height(8 bytes) + pixel data
                MAGIC_NUMBER = struct.pack(
                    ">Q", 4919426490892632400
                )  # UTF-8 encoding for 'DEPTHMAP'
                width_encoded = struct.pack(">Q", width)
                height_encoded = struct.pack(">Q", height)

                # Create the header
                depth_header = MAGIC_NUMBER + width_encoded + height_encoded

                # Combine header and raw depth data
                raw_depth_bytes = depth_header + depth_data.tobytes()

                if raw:
                    return raw_depth_bytes
                return ViamImage(raw_depth_bytes, CameraMimeType.VIAM_RAW_DEPTH)

            if mime_type == CameraMimeType.JPEG:
                depth_format = frame.get_format()
                if depth_format != OBFormat.Y16:
                    raise Exception("Depth format is not Y16")

                scale = frame.get_depth_scale()
                depth_data = np.frombuffer(frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((height, width))

                depth_data = depth_data.astype(np.float32) * scale
                depth_data = np.where(
                    (depth_data > self.MIN_DEPTH) & (depth_data < self.MAX_DEPTH),
                    depth_data,
                    0,
                )
                depth_data = depth_data.astype(np.uint16)

                depth_image = cv2.normalize(
                    depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
                pil_image = Image.fromarray(depth_image, "RGB")
                return pil_to_viam_image(pil_image, CameraMimeType.JPEG)

        if sensor == "color":
            if mime_type == CameraMimeType.JPEG:
                color_format = frame.get_format()

                try:
                    frame_data = frame.get_data()
                    data = np.asanyarray(frame_data)

                    if color_format == OBFormat.RGB:
                        rgb_data = np.resize(data, (height, width, 3))
                    elif color_format == OBFormat.BGR:
                        bgr_data = np.resize(data, (height, width, 3))
                        rgb_data = cv2.cvtColor(bgr_data, cv2.COLOR_BGR2RGB)
                    elif color_format == OBFormat.MJPG:
                        # Decode MJPG to BGR using OpenCV
                        bgr_data = cv2.imdecode(data, cv2.IMREAD_COLOR)
                        # Convert BGR to RGB for PIL
                        rgb_data = cv2.cvtColor(bgr_data, cv2.COLOR_BGR2RGB)
                    else:
                        self.logger.error(f"Unsupported color format: {color_format}")
                        return

                    pil_image = Image.fromarray(rgb_data, "RGB")
                    if raw:
                        output_buffer = BytesIO()
                        pil_image.save(output_buffer, format="JPEG")
                        rgb_bytes = output_buffer.getvalue()
                        output_buffer.close()
                        return rgb_bytes
                    return pil_to_viam_image(pil_image, CameraMimeType.JPEG)

                except Exception as err:
                    self.logger.error(f"Error converting frame to PIL Image: {err}")
                    return

        raise NotSupportedError(
            f"mime_type {mime_type} is not supported for {sensor}. Please use {CameraMimeType.JPEG} or {CameraMimeType.VIAM_RAW_DEPTH}."
        )

    async def get_images(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Tuple[List[NamedImage], ResponseMetadata]:
        if self.pipeline is None:
            self.logger.error("No device connected.")
            return

        images: List[NamedImage] = []
        # For timestamp calculation later
        timestamp: Optional[int] = None

        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return [], ResponseMetadata()

        color_frame = frames.get_color_frame()
        if color_frame is not None:
            if timestamp is None:
                timestamp = color_frame.get_timestamp_us()
            rgb_bytes = self.process_frame_as_image(
                color_frame, CameraMimeType.JPEG, raw=True, sensor="color"
            )
            images.append(NamedImage("color", rgb_bytes, CameraMimeType.JPEG))

        depth_frame = frames.get_depth_frame()
        if depth_frame is not None:
            if timestamp is None:
                timestamp = depth_frame.get_timestamp_us()
            depth_bytes = self.process_frame_as_image(
                depth_frame, CameraMimeType.VIAM_RAW_DEPTH, raw=True, sensor="depth"
            )
            images.append(
                NamedImage("depth", depth_bytes, CameraMimeType.VIAM_RAW_DEPTH)
            )

        if len(images) > 0 and timestamp is not None:
            seconds_float = timestamp / 1_000_000
            seconds = int(seconds_float)
            nanoseconds = int((seconds_float - seconds) * 1_000_000_000)
            metadata = ResponseMetadata(
                captured_at=Timestamp(seconds=seconds, nanos=nanoseconds)
            )
            return images, metadata

        return [], ResponseMetadata()

    async def get_point_cloud(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[bytes, str]:
        """
        Gets the next point cloud from the camera.

        Returns:
            Tuple[bytes, str]: The point cloud data and its mime type.
        """
        if self.pipeline is None:
            self.logger.error("No device connected.")
            return

        try:
            # Get frames from the camera
            frames = self.pipeline.wait_for_frames(100)
            if not frames:
                raise Exception("No frames received from camera")

            # Get depth and color frames
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame:
                raise Exception("No depth frame received")

            # Create and configure filters for point cloud generation
            align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            point_cloud_filter = PointCloudFilter()

            # Set the point cloud format based on whether we have color data
            has_color = color_frame is not None
            # has_color = False
            if has_color:
                point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)
            else:
                point_cloud_filter.set_create_point_format(OBFormat.POINT)

            # Align frames
            aligned_frames = align_filter.process(frames)
            if aligned_frames is None:
                raise Exception("Failed to align frames")

            # Generate point cloud
            point_cloud_frame = point_cloud_filter.process(aligned_frames)
            if point_cloud_frame is None:
                raise Exception("Failed to generate point cloud")

            # Extract points from point cloud frame
            points = point_cloud_filter.calculate(point_cloud_frame)

            # Convert to meters (divide by 1000) and to float32
            points_xyz = (np.array([p[:3] for p in points]) / 1000.0).astype(np.float32)

            # Handle colored vs non-colored point clouds
            if has_color:
                # Extract RGB colors
                colors = (
                    np.array([p[3:6] for p in points])
                    .astype(np.uint8)
                    .astype(np.uint32)
                )

                assert colors is not None
                # Pack RGB into a single float as required by PCD format
                rgb_int = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
                rgb_float = rgb_int.view(np.float32)

                # Concatenate the xyz coordinates with the packed rgb value
                colored_points = np.column_stack((points_xyz, rgb_float))

                # Create PCD header for colored point cloud
                version = "VERSION .7\n"
                fields = "FIELDS x y z rgb\n"
                size = "SIZE 4 4 4 4\n"
                type_of = "TYPE F F F F\n"
                count = "COUNT 1 1 1 1\n"
                height = "HEIGHT 1\n"
                viewpoint = "VIEWPOINT 0 0 0 1 0 0 0\n"
                width = f"WIDTH {colored_points.shape[0]}\n"
                points_count = f"POINTS {colored_points.shape[0]}\n"
                data = "DATA binary\n"
                header = f"{version}{fields}{size}{type_of}{count}{width}{height}{viewpoint}{points_count}{data}"
                header_bytes = bytes(header, "UTF-8")

                # Combine header and point cloud data
                pcd_data = header_bytes + colored_points.tobytes()
            else:
                # Create PCD header for non-colored point cloud
                version = "VERSION .7\n"
                fields = "FIELDS x y z\n"
                size = "SIZE 4 4 4\n"
                type_of = "TYPE F F F\n"
                count = "COUNT 1 1 1\n"
                height = "HEIGHT 1\n"
                viewpoint = "VIEWPOINT 0 0 0 1 0 0 0\n"
                width = f"WIDTH {points_xyz.shape[0]}\n"
                points_count = f"POINTS {points_xyz.shape[0]}\n"
                data = "DATA binary\n"
                header = f"{version}{fields}{size}{type_of}{count}{width}{height}{viewpoint}{points_count}{data}"
                header_bytes = bytes(header, "UTF-8")

                # Combine header and point cloud data
                pcd_data = header_bytes + points_xyz.tobytes()

            # Return the point cloud data and its MIME type
            return pcd_data, CameraMimeType.PCD
        except Exception as e:
            self.logger.error(f"Failed to get point cloud: {str(e)}")
            raise Exception(f"Failed to get point cloud: {str(e)}")

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Camera.Properties:
        if self.pipeline is None:
            return

        device = self.pipeline.get_device()
        if not device:
            raise Exception("No device available")

        # Get camera parameters from the device
        try:
            camera_params = self.pipeline.get_camera_param()
            intrinsics = IntrinsicParameters(
                focal_x_px=camera_params.rgbIntrinsic.fx,
                focal_y_px=camera_params.rgbIntrinsic.fy,
                center_x_px=camera_params.rgbIntrinsic.cx,
                center_y_px=camera_params.rgbIntrinsic.cy,
                width_px=camera_params.rgbIntrinsic.width,
                height_px=camera_params.rgbIntrinsic.height,
            )

            return GetPropertiesResponse(
                supports_pcd=True,
                intrinsic_parameters=intrinsics,
                distortion_parameters=None,
            )
        except Exception as e:
            # If we can't get intrinsic parameters, return empty ones
            return GetPropertiesResponse(
                supports_pcd=True,
                intrinsic_parameters=IntrinsicParameters(),
                distortion_parameters=DistortionParameters(),
            )

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        self.logger.error("`do_command` is not implemented")
        raise NotImplementedError()

    async def get_geometries(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None
    ) -> List[Geometry]:
        self.logger.error("`get_geometries` is not implemented")
        raise NotImplementedError()

    async def close(self):
        if self.pipeline is None:
            return

        self.pipeline.stop()
        self.pipeline = None
