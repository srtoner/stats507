#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# # Problem Set 6: Intro to GIT
# *Work produced by Stephen Toner, Fall 2021* \
# email: srtoner@umich.edu

# ## Question 0

# 1. Assuming you already have a github account (if not, follow instructions
# [here](https://github.com/) ), proceed to the next step.
#
# 2. To create a repository, you can either navigate through the instructions
# at github or execute the following steps:
# * Navigate to where you want your repository to be in your file system
# * `git init`
# * `git remote add origin https://github.com/[username]/[repo_name].git`
#
# I've already taken the steps to create a repo called 
# [stats507](https://github.com/srtoner/stats507). Feel free to take a look.

# ## Question 1 - Using Git
# I've added my solutions for Problem Sets 2 and 4 to the repository for
# ease of access. The code is in `507_hw2.py` and `507_hw4.py`, respectively.

# 1. The code in question is in 507_hw2.py; we can copy the module imports
# (lines 11-17) and the code itself (lines 570-685) using `sed`:
# * `sed -n '11,17p;18q' 507_hw2.py > ps2q3.py`
# * `sed -n '570,685p;686q' 507_hw2.py > cat >> ps2q3.py`
#
# We can then add the new file to the repo by executing

# 2. Suppose we want to add a README file. We would create it using vim
# (or another text editor) as follows:
# * `vim README.md`
# * press `:i` and type what you want to insert. When done, hit escape
# * press `:wq` to write and quit
# In this case, we want to add a description of what the extracted code does 
# and include a link. We can do this in Markdown with the following syntax
# * `[text to display](ps2q3.py)`

# 3. Once finished, we want to push the changes we've made to the remote repo:
# * `git add ps2q3.py README.md`
# * `git commit -m 'adding ps2q3 code and README.md'`
# * `git push`
#
# To view the specific commit, click 
# [here](https://github.com/srtoner/stats507/commit/810c289f3cc0af4219b8442faeabce8b34e82207)

# 4. Create a branch using `git branch ps4` and check it out using 
# `git checkout ps4`. From here we can modify our code in `ps2q3.py` to add 
# the gender field to the dataset. Using vim, we can make these changes by
# inserting:
# * `"RIAGENDR",` on line 20
# *  `"RIAGENDR" : "gender",` on line 33

# Write the changes to file and close. We can then commit the changes and 
# create an upstream branch as follows:
# * `git commit -a ps2q3.py -m "adding gender"`
# * `git push -u origin HEAD`
# 
# This creates an upstream branch with the same name as our current branch.
# 
# To merge with the existing main branch, we would proceed with:
# * `git checkout main`
# * `git pull`
# * `git merge ps4`
# * `git push origin main`
# 
# To view this specific commit, click 
# [here](https://github.com/srtoner/stats507/commit/96817b7d12449f3eb25d65199ca1a0675d0a821e)

# # Question 2 - GitHub Collaboration
# 1. Create a new directory with `mkdir pandas_notes`
# 2. Extract PS4 Question 0 code with the `sed` command:
# * `sed -n '3,16p;17q' hw4/507_hw4.py > pandas_notes/pd_topic_srtoner.py`
# * `sed -n '21,171l;172q' hw4/507_hw4.py > cat >> pandas_notes/pd_topic_srtoner.py`
# * Add a title "slide" using vim
# The link to the file is [here](pandas_notes/pd_topic_srtoner.py)
# 3. I am currently awaiting a link to the repo of one person above me in the 
# fanout tree.