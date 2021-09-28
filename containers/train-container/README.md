## Building a Docker image to run the NMF model training

-----
* 4/26/21 jmp - initial setup and testing

-----
### Notes

The `Dockerfile` creates an image based on the Docker `python:3.7-slim-buster` container and then installs the required module for the model training and runs the model training
`run_nmf.py`
