#!/usr/bin/env python
"""Script to build code"""
import argparse
import os
from pathlib import Path

import hu_build.build_docker
from hu_build.build_docker import DockerImage

SCRIPT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
ROOT_DIR = SCRIPT_PATH.parent


def main(build_args):
    """Main function"""
    src_path = ROOT_DIR / 'src'
    if not build_args.no_test:
        # TODO: add unit tests
        pass
    if build_args.docker_build:
        tag_version = build_args.version
        docker_image = DockerImage(
            src_path,
            'backend/embedding',
            image_tag=tag_version,
            registry='eu.gcr.io/hutoma-backend')
        hu_build.build_docker.build_single_image(
            "ai-embedding", docker_image, push=build_args.docker_push)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Python Common build command-line')
    PARSER.add_argument('--no-test', help='skip tests', action="store_true")
    PARSER.add_argument('--version', help='build version', default='latest')
    PARSER.add_argument(
        '--docker-build', help='Build docker', action="store_true")
    PARSER.add_argument(
        '--docker-push', help='Push docker images to GCR', action="store_true")
    BUILD_ARGS = PARSER.parse_args()
    main(BUILD_ARGS)
