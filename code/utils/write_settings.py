import json
import os

settings = {
    'model': {
        'subject_name': 'model',
        'subject_path': os.path.join('test_models', 'model_conv.py'),
        'mutations': ["change_loss_function",
                      "change_activation_function",
                      "add_activation_function",
                      "change_epochs",
                      "change_batch_size",
                      "change_learning_rate",
                      "remove_activation_function",
                      "add_weights_regularisation",
                      "change_dropout_rate",
                      "change_weights_initialisation",
                      "change_optimisation_function",
                      "change_patience",
                      "change_gradient_clip"]
    }
}


def write_subject_settings():
    global settings
    with open('subject_settings.json', 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)


def read_subject_settings(subject):
    with open('subject_settings.json') as data_file:
        data = json.load(data_file)

    settings = data.get(subject, None)
    return settings


if __name__ == '__main__':
    write_subject_settings()