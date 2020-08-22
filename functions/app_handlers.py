# Copyright (C) 2019 Christopher Gearhart
# chris@bblanimation.com
# http://bblanimation.com/
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# System imports
# NONE!

# Blender imports
import bpy
from bpy.app.handlers import persistent
from .property_callbacks import *

# Module imports
# NONE!


@persistent
def handle_upconversion(scn):
    # rename outdated ABS Plastic Material names
    pink_mat = bpy.data.materials.get('ABS Plastic Pink')
    if pink_mat is not None:
        pink_mat.name = 'ABS Plastic Dark Pink'
    orange_mat = bpy.data.materials.get('ABS Plastic Trans-Reddish Orange')
    if orange_mat is not None:
        orange_mat.name = 'ABS Plastic Trans-Orange'

@persistent
def verify_texture_data(scn):
    mat_names = get_mat_names()  # list of materials to append from 'abs_plastic_materials.blend'
    already_imported = [mn for mn in mat_names if bpy.data.materials.get(mn) is not None]
    if len(already_imported) > 0:
        update_fd_image(scn, bpy.context)
        update_s_image(scn, bpy.context)
