# Copyright (c) Facebook, Inc. and its affiliates.
from iopath.common.file_io import HTTPURLHandler, OneDrivePathHandler, PathHandler
from iopath.common.file_io import PathManager as PathManagerBase

__all__ = ["PathManager", "PathHandler"]


PathManager = PathManagerBase()
"""
This is a detectron2 project-specific PathManager.
We try to stay away from global PathManager in fvcore as it
introduces potential conflicts among other libraries.
"""