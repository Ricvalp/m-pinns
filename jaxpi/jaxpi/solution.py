import numpy as np
from tqdm import tqdm


def get_final_solution(charts, charts_idxs, u_preds):
    points = []
    solutions = []
    solution_idxs = []

    for key in tqdm(charts_idxs.keys()):
        for i, idx in enumerate(charts_idxs[key]):
            if idx not in solution_idxs:
                points.append(charts[key][i])
                solution_idxs.append(idx)
                solutions.append([u_preds[key][i]])
            else:
                idx_2 = solution_idxs.index(idx)
                solutions[idx_2].append(u_preds[key][i])

    final_sol = []
    for sol in solutions:
        final_sol.append(np.mean(sol, axis=0))

    points = np.array(points)
    final_sol = np.array(final_sol)

    return points, final_sol


def load_solution(solution_path):
    data = np.load(solution_path, allow_pickle=True).item()
    pts = data["pts"]
    sol = data["sol"]
    u_preds = data["u_preds"]
    mesh_sol = data["mesh_sol"]
    gt_sol = data["gt_sol"]
    return pts, sol, u_preds, mesh_sol, gt_sol


def save_solution(solution_path, pts, sol, u_preds, mesh_sol, gt_sol):
    np.save(
        solution_path,
        {
            "pts": pts,
            "sol": sol,
            "u_preds": u_preds,
            "mesh_sol": mesh_sol,
            "gt_sol": gt_sol,
        },
        allow_pickle=True,
    )
