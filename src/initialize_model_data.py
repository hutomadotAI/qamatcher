#!/usr/bin/env python
"""Script to download models"""
import spacy
import sys
import subprocess
import nltk


def pip_install(package):
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", package], check=True)


def pip_show(package):
    result = subprocess.run([sys.executable, "-m", "pip", "show", package], encoding="utf8", stdout=subprocess.PIPE)
    if result.returncode == 1:
        # Not found
        return None
    elif result.returncode != 0:
        # Some other issue
        raise Exception("pip show failed with error {}".format(result.returncode))
    
    pip_show_text = result.stdout
    
    pip_details = {}
    for line in pip_show_text.splitlines():
        line_items = line.split(':', 1)
        pip_details[line_items[0].strip()] = line_items[1].strip()
    return pip_details


def load_model(model, version):
    # It's quicker to check in PIP than load it in Spacy
    package_details = pip_show(model)
    if not package_details:
        return (False, None)
    found_version = package_details['Version']
    return (version == found_version, found_version)


def download_model(model, version):
    # Borrow from implementation of spacy's download command
    # But extend it so we can actually tell if it FAILED
    download_url = spacy.about.__download_url__ + \
        "/{m}-{v}/{m}-{v}.tar.gz".format(
            m=model, v=version)
    pip_install(download_url)


if __name__ == "__main__":
    print("*** Initialize Model script")
    LANGUAGES = [('en_core_web_sm', '2.0.0')]

    print("***********************************************************")
    print("*** Spacy models")
    for model, version in LANGUAGES:
        download_model_required = True
        print("***********************************************************")
        print("*** Checking if {}:{} is available...".format(model, version))
        version_matches, found_version = load_model(model, version)
        if not version_matches:
            print("*** Model {}:{} not present, found {}".format(
                model, version, found_version))
        else:
            print("*** {}:{} found".format(model, version))
            download_model_required = False
        if download_model_required:
            download_model(model, version)

    print("***********************************************************")
    print("*** nltk")
    nltk.downloader.download("stopwords")
    nltk.downloader.download("brown")
    print("*************************DONE******************************")
