"""Containers, all built from scratch.

The rule for this package: where a structure is the thing being demonstrated, it
does not get to use a built in Python container underneath. A dynamic array that
stores its items in a Python list is not a dynamic array, it is a wrapper around
someone else's dynamic array, and it teaches nothing about growth or copying.

So the dynamic array allocates raw memory through `ctypes`, the linked lists use
node objects with real pointers, and the hash tables do their own probing. Where
a plain list is used for something incidental (a scratch buffer, a list of
results being returned to the caller) that is fine and is noted in the code.
"""
