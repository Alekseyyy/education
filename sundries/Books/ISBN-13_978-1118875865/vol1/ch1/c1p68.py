
## Matter & Interactions, Fourth Edition; Computational Problem Sets
## Chapter 1, Problem 68
## Implemented by: A. S. "Aleksey" Ahmann, hackermaneia@riseup.net

## PROBLEM: Watch the first introductory VPython video, VPython
## Instructional Videos 1: 3D Objects, at vpython.org/video01.html and
## complete the challenge activity at the end of the video.

## SOLUTION: The challenge described in the video (starting at the
## 3:10 mark) is to recreate the three-dimensional scene (as shown
## in the 3:13 mark). I've defined a new function called 'render_this'
## which takes a position vector, an axis vector, and a name for the
## shape to be rendered, and calls upon the 'sphere' and 'arrow'
## functions to render said objects, which has its parameters stored
## in a multidimensional list.

from visual import *

def render_this(v_pos, v_aaxis, thing_name):
	print("[*] Rendering the", thing_name, "Thing...")
	sphere(pos=v_pos, radius=0.5, color=color.green)
	arrow(pos=v_pos, axis=v_aaxis, color=color.red)

parameters = [
	[vector(0, 0, 0), vector(0, -1, 0), "Middle-Centre"],
	[vector(+2, +1, 0), vector(-1, 0, 0), "Top-Right"],
	[vector(+2, -1, 0),vector(+1, +1, 0),"Bottom-Right"]
]

for n, k in enumerate(parameters):
	render_this(k[0], k[1], k[2])
