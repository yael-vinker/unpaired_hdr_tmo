import params
import os
import torch


def create_dir(result_dir_pref, model_name, con_operator, model_path, loss_graph_path, result_path, model_depth):
    # output_dir = os.path.join("/cs","labs","raananf","yael_vinker","29_07",result_dir_pref + "_" + model_name)
    output_dir = os.path.join(result_dir_pref + "_" + model_name + "_" + con_operator + "_depth_" + str(model_depth))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Directory ", output_dir, " created")

    best_acc_path = os.path.join(output_dir, params.best_acc_images_path)
    model_path = os.path.join(output_dir, model_path)
    models_250_save_path = os.path.join("models_250", "models_250_net.pth")
    model_path_250 = os.path.join(output_dir, models_250_save_path)
    best_model_save_path = os.path.join("best_model", "best_model.pth")
    best_model_path = os.path.join(output_dir, best_model_save_path)
    loss_graph_path = os.path.join(output_dir, loss_graph_path)
    result_path = os.path.join(output_dir, result_path)
    acc_path = os.path.join(output_dir, "accuracy")
    tmqi_path = os.path.join(output_dir, "tmqi")
    gradient_flow_path = os.path.join(output_dir, params.gradient_flow_path, "g")

    if not os.path.exists(best_acc_path):
        os.mkdir(best_acc_path)
        print("Directory ", best_acc_path, " created")

    if not os.path.exists(os.path.dirname(gradient_flow_path)):
        os.makedirs(os.path.dirname(gradient_flow_path))
        print("Directory ", gradient_flow_path, " created")

    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
        print("Directory ", model_path, " created")

    if not os.path.exists(os.path.dirname(model_path_250)):
        os.makedirs(os.path.dirname(model_path_250))
        print("Directory ", model_path_250, " created")

    if not os.path.exists(os.path.dirname(best_model_path)):
        os.makedirs(os.path.dirname(best_model_path))
        print("Directory ", best_model_path, " created")

    if not os.path.exists(loss_graph_path):
        os.mkdir(loss_graph_path)

        print("Directory ", loss_graph_path, " created")
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        print("Directory ", result_path, " created")
    if not os.path.exists(acc_path):
        os.mkdir(acc_path)
        print("Directory ", acc_path, " created")

    if not os.path.exists(tmqi_path):
        os.mkdir(tmqi_path)
        print("Directory ", tmqi_path, " created")
    return output_dir



if __name__ == '__main__':
    import utils.image_quality_assessment_util as utils_
    utils_.calculate_TMQI_results_for_selected_methods("/cs/labs/raananf/yael_vinker/data/test/tmqi_test_hdr", "/cs/labs/raananf/yael_vinker/NEW_TMQI")