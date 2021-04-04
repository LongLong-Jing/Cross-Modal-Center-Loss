def calculate_accuracy(outputs, targets):
    batch_size = outputs.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()

    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return 1.0 * n_correct_elems / batch_size
