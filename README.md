# BlendiPose

This is a Python library for rendering human motion videos, mainly to visualize the output of 3D human pose and shape estimation models. It's similar to my other project [PoseViz](https://github.com/isarandi/poseviz), but optimized for beautiful rendering quality instead of speed.

BlendiPose is built on top of Blender, and the [Blendify](https://github.com/ptrvilya/blendify) package (specifically my fork of it, which is vendored in this repository).

## References

Regarding Blendify, see: 

```bibtex
@article{blendify2024,
  title={Blendify -- Python rendering framework for Blender},
  author={Guzov, Vladimir and Petrov, Ilya A and Pons-Moll, Gerard},
  journal={arXiv preprint arXiv:2410.17858},
  year={2024}
}
```