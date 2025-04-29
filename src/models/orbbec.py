from io import BytesIO
import sys
import tempfile
from typing import Any, ClassVar, Dict, Final, List, Mapping, Optional, Sequence, Tuple

import open3d as o3d
import cv2
import numpy as np
from PIL import Image
from typing_extensions import Self
from viam.components.camera import Camera
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
from viam.utils import ValueTypes

try:
    import pyorbbecsdk
    from pyorbbecsdk import (
        AlignFilter,
        Pipeline,
        Config,
        Context,
        OBError,
        OBFormat,
        OBSensorType,
        OBStreamType,
        PointCloudFilter,
    )

    print(f"pyorbbecsdk version: {pyorbbecsdk.get_version()}")
except ImportError:
    print("Error: pyorbbecsdk is not installed or cannot be imported.")
    print("Please run: ./setup.sh")
    sys.exit(1)


class Orbbec(Camera, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "camera"), "orbbec")

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
        return []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both implicit and explicit)
        """
        context = Context()
        device_list = context.query_devices()

        device_count = device_list.get_count()
        if device_count == 0:
            raise Exception(
                "No Orbbec devices found. Please connect a device and try again."
            )

        self.logger.info(f"Found {device_count} Orbbec device(s)")
        device = device_list.get_device_by_index(0)
        device_info = device.get_device_info()
        self.logger.info(
            f"Device name: {device_info.get_name()}, PID: {device_info.get_pid()}, Serial Number: {device_info.get_serial_number()}"
        )
        self.pipeline = Pipeline(device)
        self.config = Config()

        color_profile_list = self.pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
        self.logger.info(f"Available color profiles: {len(color_profile_list)}")
        try:
            color_profile = color_profile_list.get_video_stream_profile(
                0, 720, OBFormat.RGB, 30
            )
        except OBError as err:
            color_profile = color_profile_list.get_default_video_stream_profile()
            self.logger.warn(f"Unable to set RGB format, using {color_profile}")
        self.config.enable_stream(color_profile)

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
        self.logger.info(f"get_image - mime_type: {mime_type}")
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return
        color_frame = frames.get_color_frame()

        if color_frame is None:
            return

        return self.process_frame_as_image(color_frame)

    def process_frame_as_image(self, frame) -> ViamImage:
        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()

        try:
            frame_data = frame.get_data()
            self.logger.info(f"frame_data - type {type(frame_data)}")
            self.logger.info(f"frame_data std_dev: {np.std(frame_data)}")
            data = np.asanyarray(frame_data)
            self.logger.info(f"Frame data: {data.shape} as type {data.dtype}")
            self.logger.info(f"Frame data len: {len(data)}")
            self.logger.info(
                f"Frame format: {color_format}, width: {width}, height: {height}"
            )
            self.logger.info(f"Frame data std_dev: {np.std(data)}")

            if color_format == OBFormat.RGB:
                rgb_data = np.resize(data, (height, width, 3))
                self.logger.info(
                    f"RGB data - shape: {rgb_data.shape}, as type {rgb_data.dtype}, len: {len(rgb_data)}"
                )

                pil_image = Image.fromarray(rgb_data, "RGB")
                self.logger.info(f"pil_image - {pil_image.width}x{pil_image.height}")
                image_array = np.array(pil_image)
                if image_array.size > 0:
                    std_dev = np.std(image_array)
                    if std_dev < 5.0:
                        self.logger.warn(
                            f"Image appears very uniform (std_dev: {std_dev})"
                        )
                buffer = BytesIO()
                pil_image.save(buffer, format="JPEG")

                raw_bytes = buffer.getvalue()
                self.logger.info(f"raw_byes - size {len(raw_bytes)}")
                buffer.close()
                return ViamImage(raw_bytes, CameraMimeType.JPEG)
            elif color_format == OBFormat.BGR:
                bgr_data = np.resize(data, (height, width, 3))
                rgb_data = cv2.cvtColor(bgr_data, cv2.COLOR_BGR2RGB)
                return pil_to_viam_image(
                    Image.fromarray(rgb_data, "RGB"), CameraMimeType.VIAM_RGBA
                )
            elif color_format == OBFormat.MJPG:
                # Decode MJPG to BGR using OpenCV
                bgr_data = cv2.imdecode(data, cv2.IMREAD_COLOR)
                self.logger.info(f"bgr_data: {len(bgr_data)}")
                # Convert BGR to RGB for PIL
                rgb_data = cv2.cvtColor(bgr_data, cv2.COLOR_BGR2RGB)
                return pil_to_viam_image(
                    Image.fromarray(rgb_data, "RGBA"), CameraMimeType.VIAM_RGBA
                )
            else:
                self.logger.error(f"Unsupported color format: {color_format}")
                return

        except Exception as err:
            self.logger.error(f"Error converting frame to PIL Image: {err}")
            return

    async def get_images(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Tuple[List[NamedImage], ResponseMetadata]:
        self.logger.error("`get_images` is not implemented")
        pass

    async def get_point_cloud(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[bytes, str]:
        return [], CameraMimeType.PCD
        # try:
        #     # Get frames from the camera
        #     frames = self.pipeline.wait_for_frames(100)
        #     if not frames:
        #         raise Exception("No frames received from camera")

        #     # Get depth and color frames
        #     depth_frame = frames.get_depth_frame()
        #     color_frame = frames.get_color_frame()

        #     if not depth_frame:
        #         raise Exception("No depth frame received")

        #     # Create and configure filters for point cloud generation
        #     align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        #     point_cloud_filter = PointCloudFilter()

        #     # Set the point cloud format based on whether we have color data
        #     if color_frame is not None:
        #         point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)
        #     else:
        #         point_cloud_filter.set_create_point_format(OBFormat.POINT)

        #     # Align frames
        #     aligned_frames = align_filter.process(frames)
        #     if aligned_frames is None:
        #         raise Exception("Failed to align frames")

        #     # Generate point cloud
        #     point_cloud_frame = point_cloud_filter.process(aligned_frames)
        #     if point_cloud_frame is None:
        #         raise Exception("Failed to generate point cloud")

        #     # Extract points from point cloud frame
        #     points = point_cloud_filter.calculate(point_cloud_frame)

        #     # Create Open3D point cloud
        #     pcd = o3d.geometry.PointCloud()

        #     if color_frame is not None:
        #         # Extract points and colors for RGB point cloud
        #         points_array = np.array([p[:3] for p in points])  # XYZ points
        #         colors_array = np.array([p[3:6] for p in points])  # RGB colors

        #         # Add points and colors to Open3D point cloud
        #         pcd.points = o3d.utility.Vector3dVector(points_array)
        #         pcd.colors = o3d.utility.Vector3dVector(
        #             colors_array / 255.0
        #         )  # Normalize colors to [0, 1]
        #     else:
        #         # Extract points for non-colored point cloud
        #         points_array = np.array([p[:3] for p in points])  # XYZ points

        #         # Add points to Open3D point cloud
        #         pcd.points = o3d.utility.Vector3dVector(points_array)

        #     # Write the point cloud to PCD format in memory
        #     # with tempfile.NamedTemporaryFile(delete=True) as tmp:
        #     #     o3d.io.write_point_cloud(tmp.name, pcd, write_ascii=True)
        #     #     tmp.flush()
        #     #     tmp.seek(0)
        #     #     pcd_data = tmp.read()
        #     pcd_data = BytesIO().getvalue()

        #     # Return the point cloud data and its MIME type
        #     return pcd_data, CameraMimeType.PCD
        # except Exception as e:
        #     raise Exception(f"Failed to get point cloud: {str(e)}")

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Camera.Properties:
        device = self.pipeline.get_device()
        if not device:
            raise Exception("No device available")

        # Get camera parameters from the device
        try:
            camera_params = self.pipeline.get_camera_param()
            intrinsics = IntrinsicParameters(
                fx=camera_params.rgbIntrinsic.fx,
                fy=camera_params.rgbIntrinsic.fy,
                ppx=camera_params.rgbIntrinsic.cx,
                ppy=camera_params.rgbIntrinsic.cy,
                width=camera_params.rgbIntrinsic.width,
                height=camera_params.rgbIntrinsic.height,
            )

            distortion = DistortionParameters(
                k1=camera_params.rgbDistortion.k1,
                k2=camera_params.rgbDistortion.k2,
                p1=camera_params.rgbDistortion.p1,
                p2=camera_params.rgbDistortion.p2,
                k3=camera_params.rgbDistortion.k3,
            )

            return GetPropertiesResponse(
                supports_pcd=True,
                intrinsic_parameters=intrinsics,
                distortion_parameters=distortion,
                mime_types=["image/jpeg", "pointcloud/pcd"],
            )
        except Exception as e:
            # If we can't get intrinsic parameters, return empty ones
            return GetPropertiesResponse(
                supports_pcd=True,
                intrinsic_parameters=IntrinsicParameters(),
                distortion_parameters=DistortionParameters(),
                mime_types=["image/jpeg", "pointcloud/pcd"],
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
        self.pipeline.stop()
