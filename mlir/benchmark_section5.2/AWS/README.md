# How to replicate

- We use an AWS instance: c5.12xlarge (Intel(R) Xeon(R) Platinum 8275CL CPU @ 3.00GHz)

- Once the instance is running, ssh to it and install docker. Then type:

1) docker pull lchelini/cgo
2) docker run -it lchelini/cgo
3) ./build.sh
4) ./experiment5.X.sh where X is 1, 2 or 3.
