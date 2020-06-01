import datetime
import json
import sys

def load_param_data(params_dir = "../params/",
                    model_params_file = "cnn_model.json",
                    general_params_file = "general.json"):

    """
    load params from the files
    """
    # we'll store data in a params_dict
    params_dict = {}
    # Load the params
    with open(params_dir + model_params_file, 'r') as f:
        data = f.read()
    params_dict["model_params"] = json.loads(data)
    params_dict["model_params"]["model_param_file"] = model_params_file
    with open(params_dir + general_params_file, 'r') as f:
        data = f.read()
    params_dict["general_params"] = json.loads(data)
    params_dict["general_params"]["general_param_file"] = general_params_file
    print("loaded params dict")
    print(json.dumps(params_dict, indent=4, sort_keys=True))
    return params_dict

def create_results_folder(params_dict,
                          results_dir="../data/trained_models/",
                          results_mapper_file="../data/trained_models/mapper.txt"):
    """
    Create a new folder to store the results
    based on the params and time! Also, have an additional
    json file which stores the params dict and the corresponding
    file name. This way we can note where the resutls are coming from.
    """
    import uuid
    import os
    # create a unique name
    # we'll use a uuid to generate a unique id
    # if you want to hash use - hash(frozenset(params_dict.items()))
    dir_name = results_dir + str(uuid.uuid4())[:8] + "_" +\
                    datetime.datetime.utcnow(\
                    ).strftime("%Y%m%d-%H:%M:%S")
    # Create target Directory if don't exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        print("Directory " + dir_name + " created ")
    else:    
        print("Directory " + dir_name + " already exists")
        print("check again before proceeding!")
        return dir_name
    # store the params_dict in a file
    out_param_file = dir_name + '/params.json'
    with open(out_param_file, 'w') as f:
        json.dump(params_dict, f)
    # store the data in the results mapper file    
    # check if the file exists, if yes read its contents
    # and add new details and save it.
    # else just store new info
    if os.path.exists(results_mapper_file):
        with open(results_mapper_file, 'r') as f:
            data = f.read()
            if len(data) > 0:
                feeds = json.loads(data)
                feeds[dir_name] = params_dict
            else:
                feeds = {}
                feeds[dir_name] = params_dict
        f.close()
        with open(results_mapper_file, 'w') as f:
            json.dump(feeds, f)
    else:
        with open(results_mapper_file, 'w') as f:
            feeds = {}
            feeds[dir_name] = params_dict
            json.dump(feeds, f)
    return dir_name
    
if __name__ == "__main__":

    params_dir = "../params/"
    model_params_file = "cnn_model.json"
    general_params_file = "general.json"
    results_dir="../data/trained_models/"
    results_mapper_file="../data/trained_models/mapper.txt"

    # load the parameters dict
    par_dict = load_param_data(params_dir=params_dir,
                               model_params_file=model_params_file,
                               general_params_file=general_params_file)

    # create a results folder
    results_dir = create_results_folder(par_dict, results_dir=results_dir,
                                        results_mapper_file=results_mapper_file)

