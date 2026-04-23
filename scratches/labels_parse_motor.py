import csv


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

    return parse_run(run_lines_list[-1])


def parse_time(line):
    return int(line.split(",")[0])


def parse_run(run_lines):
    """"""

    """image_ids = find_matching(run_lines, "image")
    btn_events = find_matching(run_lines, "bouton_")
    trigger_events = find_matching(run_lines, "trigger")"""

    sync = find_matching(run_lines, "Synchro_IRM")
    global_t_start = parse_time(run_lines[sync[0]])


    presented_croix = find_matching(run_lines, "image croix")
    croix_list = [run_lines[start:end] for start, end in zip(presented_croix, presented_croix[1:])]

    motor_run = [['response', 'response time']]

    for croix in croix_list:
        response = find_matching(croix, "image response")
        btn = find_matching(croix, "bouton_")

        if not response:
            continue

        resp_time = parse_time(croix[response[0]])

        resp = 1 if btn else 0

        motor_run.append([resp, resp_time - global_t_start])

    return motor_run

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
    motor = parse(i)

    with open(f"../labels/motor/labels_{i}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(motor)
