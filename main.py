#!/usr/bin/python

import os
import click

from synthnn import SynthNN

def main():
    print "SynthNN v0.1b\n"

    network = SynthNN()
    network.plot()
    
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    main()
