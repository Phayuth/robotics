def print_dict(dict):
    for sub in dict:
        print(sub, ':', dict[sub])

if __name__ == "__main__":
    perfMatrix = {
            "totalPlanningTime": 0.0,
            "KCDTimeSpend": 0.0,
            "planningTimeOnly": 0.0,
            "numberOfKCD": 0,
            "avgKCDTime": 0.0,
            "numberOfNodeTreeStart": 0,
            "numberOfNodeTreeGoal" : 0,
            "numberOfNode": 0,
            "numberOfMaxIteration": 0,
            "numberOfIterationUsed": 0,
            "searchPathTime": 0.0,
            "numberOfPath" : 0,
            "numberOfPathPruned": 0
        }
    
    print_dict(perfMatrix)