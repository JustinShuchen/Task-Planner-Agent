task_type_dict = {
    1:"organize_toolbox_one",
    2:"organize_toolbox_two",
    3:"organize_toolbox_three",
    4:"change_battery_one",
    5:"change_batteries_two",
    6:"change_batteries_four",
    7:"charging",
    8:"tighten_screw",
    9:"measure_size",
    10:"clean_euqipment"
}

#instruction: Organize tool Box; 3 类 不同数量的目标

#class: toolbox_1 (toolbox_1 1 object container close)
#class: toolbox_11 (toolbox_1 1 object container open)
#class: toolbox_2 (toolbox_2 2 objects container close)
#class: toolbox_21 (toolbox_2 2 objects container open)
#class: toolbox_3 (toolbox_3 3 objects container close)
#class: toolbox_31 (toolbox_3 3 objects container open)

def organize_toolbox_one(tool, toolbox, toolbox_open=True):
    actions = []
    if toolbox_open:
        actions = [
            f"Pick up the {tool}",
            f"Put the {tool} in the {toolbox}",
            f"Close the {toolbox}",
        ]
    else:
        actions = [
            f"Open the {toolbox}",
            f"Pick up the {tool}",
            f"Put the {tool} in the {toolbox}",
            f"Close the {toolbox}",
        ]
    return actions

def organize_toolbox_two(tool1, tool2, toolbox, toolbox_open=True):
    actions = []
    if toolbox_open:
        actions = [
            f"Pick up the {tool1}",
            f"Put the {tool1} in the {toolbox}",
            f"Pick up the {tool2}",
            f"Put the {tool2} in the {toolbox}",
            f"Close the {toolbox}",
        ]
    else:
        actions = [
            f"Open the {toolbox}",
            f"Pick up the {tool1}",
            f"Put the {tool1} in the {toolbox}",
            f"Pick up the {tool2}",
            f"Put the {tool2} in the {toolbox}",
            f"Close the {toolbox}",
        ]
    return actions

def organize_toolbox_three(tool1, tool2, tool3, toolbox, toolbox_open=True):
    actions = []
    if toolbox_open:
        actions = [
            f"Pick up the {tool1}",
            f"Put the {tool1} in the {toolbox}",
            f"Pick up the {tool2}",
            f"Put the {tool2} in the {toolbox}",
            f"Pick up the {tool3}",
            f"Put the {tool3} in the {toolbox}",
            f"Close the {toolbox}",
        ]
    else:
        actions = [
            f"Open the toolbox",
            f"Pick up the {tool1}",
            f"Put the {tool1} in the {toolbox}",
            f"Pick up the {tool2}",
            f"Put the {tool2} in the {toolbox}",
            f"Pick up the {tool3}",
            f"Put the {tool3} in the {toolbox}",
            f"Close the {toolbox}",
        ]
    return actions

#instruction: Change Batteries; 3 类 不同数量的目标

#class: battery_1 (battery_toolbox_1 1 battery container close)
#class: battery__11 (battery_toolbox_1 1 battery container open)
#class: battery_toolbox_2 (battery_toolbox_2 2 batteries container close)
#class: battery_toolbox_21 (battery_toolbox_2 2 batteries container open)
#class: battery_toolbox_3 (battery_toolbox_3 3 batteries container close)
#class: battery_toolbox_31 (battery_toolbox_3 3 batteries container open)

def change_battery_one(battery, container, container_open=True):
    actions = []
    if container_open:
        actions = [
            f"Pick up the battery",
            f"Insert the battery",
        ]
    else:
        actions = [
            f"Open the battery {container}",
            f"Pick up the battery",
            f"Insert the battery",
            f"Close the battery {container}",
        ]
    return actions

def change_battery_two(battery_1, battery_2, container, container_open=True):
    actions = []
    if container_open:
        actions = [
            f"Pick uo the {battery_1}",
            f"Insert the {battery_1}",
            f"Pick up the {battery_2}",
            f"Insert the {battery_2}",
        ]
    else:
        actions = [
            f"Open the battery {container}",
            f"Pick up the {battery_1}",
            f"Insert the {battery_1}",
            f"Pick up the {battery_2}",
            f"Insert the {battery_2}",
            f"Close the battery {container}",
        ]
    return actions

def change_battery_four(battery_1, battery_2, battery_3, battery_4, container, container_open=True):
    actions = []
    if container_open:
        actions = [
            f"Pick up the {battery_1}",
            f"Insert the {battery_1}",
            f"Pick up the {battery_2}",
            f"Insert the {battery_2}",
            f"Pick up the {battery_3}",
            f"Insert the {battery_3}",
            f"Pick up the {battery_4}",
            f"Insert the {battery_4}",
        ]
    else:
        actions = [
            f"Open the battery {container}",
            f"Pick up the {battery_1}",
            f"Insert the {battery_1}",
            f"Pick up the {battery_2}",
            f"Insert the {battery_2}",
            f"Pick up the {battery_3}",
            f"Insert the  {battery_3}",
            f"Pick up the {battery_4}",
            f"Insert the {battery_4}",
            f"Close the battery {container}",
        ]
    return actions

#Instruction: Charging; 充电操作
#class: charging(charging_toolbox charger place empty)


def charging(object, place, empty=True):
    actions = []
    if empty:
        actions = [
            f"Pick up the {object}",
            f"Put the {object} on the {place}",
        ]
    else:
        actions = [
            f"Find the {object}",
            f"Pick up the {object}",
            f"Put the {object} on the {place}",
        ]
    return actions

#instruction: Tightening Screws; 紧固螺丝操作

#class: tightening_screws (nut empty)

def tighten_screw(screw, nut, nut_empty=True):
    actions = []
    if nut_empty:
        actions = [
            f"Find the {nut}",
            f"Pick up the {nut}",
            f"Tighten the screw",
        ]
    else:
        actions = [
            f"Pick up the {nut}",
            f"Tighten the screw",
        ]
    return actions

#instruction: Measuring Size; 测量大小操作

#class: measuring_size (measuring_toolbox tool empty)


def measure_size(tool, ruler, ruler_empty=True):
    actions = []
    if ruler_empty:
        actions = [
            f"Find the {ruler}",
            f"Pick up the {ruler}",
            f"Measure the size",
        ]
    else:
        actions = [
            f"Pick up the {ruler}",
            f"Measure the size",
        ]
    return actions

#instruction: Cleaning Equipment; 清洁工具操作

#class: cleaning_equipment (cleaning_toolbox tool empty)

def clean_equipment(cloth, cloth_empty=True):
    actions = []
    if cloth_empty:
        actions = [
            f"Find the {cloth}",
            f"Pick up the {cloth}",
            f"Clean the {object}",
        ]
    else:
        actions = [
            f"Pick up the {cloth}",
            f"Clean the {object}",
        ]
    return actions
