# Third-party notices & attribution

[← Back to DepthCloud](README.md)

DepthCloud was created by **Shiwam Shorya Sharma**. Its original application code, integration logic, and documentation are covered by the root [MIT license](LICENSE). MIT requires preservation of the copyright and permission notice in copies or substantial portions of that work; see the [standard MIT text](https://opensource.org/license/mit).

The root license applies only to material the DepthCloud authors are entitled to license. It does not replace licenses for vendored code, dependencies, model weights, datasets, or assets. A license for the code does not automatically grant rights to every model or input used with it. No upstream endorsement is implied.

## Bundled Apple Depth Pro source

| Item | Provenance |
| :--- | :--- |
| Upstream | [Apple Depth Pro](https://github.com/apple/ml-depth-pro) |
| Pinned revision | `9e65e4dbe9568d23c546fcec53302b10445e109e` |
| Bundled directory | [backend/vendor/ml-depth-pro](backend/vendor/ml-depth-pro) |
| Governing notice | [Apple LICENSE](backend/vendor/ml-depth-pro/LICENSE) |
| Included subcomponents | [Upstream ACKNOWLEDGEMENTS.md](backend/vendor/ml-depth-pro/ACKNOWLEDGEMENTS.md) |
| Local packaging changes | [DEPTHCLOUD_PATCH.md](backend/vendor/ml-depth-pro/DEPTHCLOUD_PATCH.md) |

Copyright (C) 2024 Apple Inc. All Rights Reserved.

The bundled source retains Apple's license and its upstream acknowledgements, including the notices for timm and DINOv2. The local adaptation changes the package version and NumPy requirement; inference source is unchanged. Apple's terms include notice, disclaimer, and endorsement provisions and do not grant additional implied rights. Read the retained license for the actual conditions.

The upstream example images remain in the vendor tree under their existing upstream context; they are not DepthCloud scan results or claims of authorship.

## Model weights: obtained separately

| Model | Source and terms | Distribution in this repository |
| :--- | :--- | :--- |
| Depth Pro | The [upstream README](https://github.com/apple/ml-depth-pro#license) states that code and weights use its [Apple license](backend/vendor/ml-depth-pro/LICENSE) | Weights excluded; loaded from local `depth_pro.pt` |
| DETR ResNet-50 | The [published model card](https://huggingface.co/facebook/detr-resnet-50) labels this checkpoint Apache-2.0; also consult [Meta's DETR repository](https://github.com/facebookresearch/detr) | Weights and model configuration excluded; supplied locally |

Local checkpoint provenance/checksums have not been independently matched against official releases. Obtain trusted weights and retain the license, notices, and any applicable model/dataset conditions supplied with the exact version you use. DepthCloud's MIT license cannot grant broader rights over them.

## Dependency credits

DepthCloud depends on these projects and their contributors:

| Projects | Role |
| :--- | :--- |
| [PyTorch](https://github.com/pytorch/pytorch), [torchvision](https://github.com/pytorch/vision) | Model execution and vision runtime |
| [Transformers](https://github.com/huggingface/transformers), [timm](https://github.com/huggingface/pytorch-image-models) | Local detection and model components |
| [COLMAP / pycolmap](https://github.com/colmap/colmap) | Camera reconstruction and bundle adjustment |
| [Open3D](https://github.com/isl-org/Open3D), [trimesh](https://github.com/mikedh/trimesh) | Geometry processing and mesh tooling |
| [OpenCV](https://github.com/opencv/opencv), [NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy) | Capture, image processing, and numerical work |
| [FastAPI](https://github.com/fastapi/fastapi), [Uvicorn](https://github.com/encode/uvicorn), [Pydantic](https://github.com/pydantic/pydantic) | Validated local API |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | Read-only tool integration |
| [React](https://github.com/facebook/react), [Three.js](https://github.com/mrdoob/three.js), [React Three Fiber](https://github.com/pmndrs/react-three-fiber) | Browser UI and 3D viewing |
| [Vite](https://github.com/vitejs/vite), [TypeScript](https://github.com/microsoft/TypeScript), [Lucide](https://github.com/lucide-icons/lucide) | Frontend tooling and icons |

This is a credit overview, not an exhaustive license inventory for every transitive package or native binary. Exact application package versions are recorded in [backend/requirements.txt](backend/requirements.txt) and [app/package-lock.json](app/package-lock.json). Their own distributions contain the applicable license texts and notices. Runtime installers, CUDA components, and model assets can have separate terms.

When shipping a combined binary, installer, frontend bundle, or model package, retain all applicable upstream notices and satisfy the terms for the artifacts actually included. The [Code of Conduct](CODE_OF_CONDUCT.md) governs participation; it does not replace these licenses or impose extra conditions on MIT use.
