from .CSA import CSA


def create_model(opt):
    print(opt.model)
    if opt.model == 'csa_net':
        model = CSA()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
