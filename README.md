# Module orbbec-camera 

Use [Orbbec cameras](https://www.orbbec.com/products/) compatible with [v2 of the Orbbec SDK](https://github.com/orbbec/pyorbbecsdk/tree/v2-main?tab=readme-ov-file#hardware-products-supported-by-python-sdk).
This module provides access to the color and depth sensors.

## Model viam-labs:camera:orbbec

Use [Orbbec cameras](https://www.orbbec.com/products/) compatible with [v2 of the Orbbec SDK](https://github.com/orbbec/pyorbbecsdk/tree/v2-main?tab=readme-ov-file#hardware-products-supported-by-python-sdk).

### Configuration
The following attribute template can be used to configure this model:

```json
{
"sensors": ["color", "depth"]
}
```

#### Attributes

The following attributes are available for this model:

| Name          | Type   | Inclusion | Description                |
|---------------|--------|-----------|----------------------------|
| `sensors` | array  | Optional  | An array that contains the strings `color` and/or `depth`, defaults to include both. The sensor that comes first in the array is designated the "main sensor" and will be the image that gets returned by `get_image` calls and what will appear in the Control tab on the [Viam app](https://app.viam.com). When both sensors are requested, `get_point_cloud` will be available for use, and `get_images` will return both the color and depth outputs. Additionally, color and depth outputs returned together will always be aligned, have the same height and width, and have the same timestamp. See [Viam's documentation on the Camera API](https://docs.viam.com/components/camera/#api) for more details. |

#### Example Configuration

```json
{
  "sensors": ["color"]
}
```

## Troubleshooting

**Unable to connect to camera**

The `first_run.sh` script included in this module should automatically install the `udev` rules for connecting to the camera on Linux devices.
If there is an issue, try copying `99-obsensor-libusb.rules` in the `scripts/` directory of this repo to `/etc/udev/rules.d/` on the Viam machine and calling the following command on the system:

```
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Development

This module depends on the [v2-main branch of the pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk/tree/v2-main) as a git submodule. So use the `--recursive` flag when cloning this repo or `git submodule update --init --recursive` if you've already cloned it.

To build this module on the target device (aarch64 or amd64 Linux machines), run `make archive.tar.gz`.

To build this module on another platform for a target device, use a tool like [canon](https://github.com/viamrobotics/canon) to run the Make command in the appropriate Docker container: `canon make archive.tar.gz`
