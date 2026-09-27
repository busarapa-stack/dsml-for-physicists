#!/usr/bin/env bash
# Ch.2  code:github-push -- REFERENCE ONLY, do not run as is.
# Replace <user> with your GitHub user name and create an empty repository
# called physics-ch02 on github.com first.
git remote add origin https://github.com/<user>/physics-ch02.git
git branch -M main
git push -u origin main

# later pushes
git push

# someone else gets a full copy
git clone https://github.com/<user>/physics-ch02.git
