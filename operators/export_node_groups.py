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
import os
import time

# Blender imports
import bpy
from bpy.types import Operator
from mathutils import Matrix, Vector

# Module imports
from ..functions import *


class ABS_OT_export_node_groups(Operator):
    """Export ABS Plastic Materials node groups to 'node_groups_2-7/8.blend' library file"""
    bl_idname = "abs.export_node_groups"
    bl_label = "Export ABS Node Groups"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(self, context):
        return bpy.props.abs_developer_mode != 0

    def execute(self, context):
        data_blocks = []
        data_blocks.append(bpy.data.images.get("ABS Fingerprints and Dust"))

        # append node groups from nodeDirectory
        group_names = ("ABS_Bump", "ABS_Dialectric", "ABS_Transparent", "ABS_Uniform Scale", "ABS_Translate")
        for group_name in group_names:
            data_blocks.append(bpy.data.node_groups.get(group_name))

        blendlib_name = "node_groups_2-8.blend" if b280() else "node_groups_2-7.blend"

        storagePath = os.path.join("/", "Users", "cgear13", "scripts", "my_scripts", "abs-plastic-materials", "lib", blendlib_name)

        bpy.data.libraries.write(storagePath, set(data_blocks), fake_user=True)

        return {"FINISHED"}
