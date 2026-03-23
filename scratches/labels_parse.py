

def find_matching(lines, reg):
    indices = []

    for i, l in enumerate(lines):
        if l.find(reg) != -1:
            indices.append(i)

    return indices


column_names = ["run", "trial", "global time", "run time", "morph level", "couple", "response", "response time"]


def parse(sub_id):

    lines = open(f"../labels/raw/labels_{sub_id}.csv").readlines()

    start_run_ids = find_matching(lines, "Debut_run")
    end_run_ids = find_matching(lines, "Fin_run")

    run_lines_list = [lines[start:end] for start, end in zip(start_run_ids, end_run_ids)]
    runs = []

    for run_lines in run_lines_list:
        parsed = parse_run(run_lines)
        runs += parsed

    return runs


def parse_time(line):
    return int(line.split(",")[0])


def parse_run(run_lines):
    """"""

    """image_ids = find_matching(run_lines, "image")
    btn_events = find_matching(run_lines, "bouton_")
    trigger_events = find_matching(run_lines, "trigger")"""

    presented_morphs = find_matching(run_lines, "MORPH")
    morph_list = [run_lines[start:end] for start, end in zip(presented_morphs, presented_morphs[1:])]
    if presented_morphs:
        morph_list += [run_lines[presented_morphs[-1]:]]

    sync = find_matching(run_lines, "Synchro_IRM")
    global_t_start = parse_time(run_lines[sync[0]])

    parsed = []

    for morph in morph_list:

        morph_line = morph[0]
        morph_data = morph_line.split(",")[3].split("\\")[-1].split(".")[0]
        morph_time = parse_time(morph_line)

        morph_level = morph_data.split("_")[1]
        morph_couple = morph_data.split("_")[2]

        response = 0
        response_time = ""

        boutons_parsed = find_matching(morph, "bouton_1")

        if boutons_parsed:
            response = 1
            btn_line = morph[boutons_parsed[-1]]

            btn_data = btn_line.split(",")[3]
            response_time = int(btn_data.split(" ")[1]) + 500

        parsed_line = [morph_time - global_t_start, morph_level, morph_couple, response, response_time]
        parsed.append(parsed_line)

    return parsed

# out:
# r = parse(1)


def cmp(sub_id):

    regen = parse(sub_id)
    origin = open(f"../labels/labels_{sub_id}.csv").readlines()[1:]

    failed = 0

    if failed:
        print(regen[failed])
        print(origin[failed])

    exclusion_list = []

    # assert len(origin) == len(regen), f"{len(origin)} != {len(regen)}, sub_id={sub_id}"
    for i in range(min(len(regen), len(origin))):
        if i in exclusion_list:
            continue

        origin_l = origin[i].strip().split(",")
        # print(origin_l)
        regen_l = regen[i]

        assert abs(int(origin_l[3]) - int(regen_l[0])) < 30, f"{origin_l[3]} != {regen_l[0]}, {i}"
        assert origin_l[4] == regen_l[1], f"{origin_l[4]} != {regen_l[1]}, {i}"
        assert origin_l[5] == regen_l[2], f"{origin_l[5]} != {regen_l[2]}, {i}"
        assert int(origin_l[6]) == int(regen_l[3]), f"{origin_l[6]} != {regen_l[3]}, {i}"
        if origin_l[7]:
            # assert int(origin_l[7]) == int(regen_l[4]), f"{origin_l[7]} != {regen_l[4]}, {i}"
            ...
        else:
            assert not origin_l[7], f"{i}"
            assert not regen_l[4], f"{regen_l[4]}, {i}"



for i in range(1, 34):
    # print(i, open(f"../labels/raw/labels_{i}.csv").readlines()[0])
    cmp(i)
