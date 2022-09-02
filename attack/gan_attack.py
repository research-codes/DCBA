import copy
import torch


def attach_ganattack_to_client(
    cls,
    target_label,
    generator,
    generator_optimizer,
    generator_criterion,

    nz=100,
    device="cpu",
    
    gan_batch_size=64,
    gan_epoch=1,
    gan_log_interval=0,
    ignore_first_download=False,
):
    class GANAttackClientWrapper(cls):

        def __init__(self, *args, **kwargs):
            super(GANAttackClientWrapper, self).__init__(*args, **kwargs)

            self.target_label = target_label
            self.generator = generator
            self.generator_optimizer = generator_optimizer
            self.generator_criterion = generator_criterion
            self.nz = nz
            self.device = device

            self.discriminator = copy.deepcopy(self.model)
            self.discriminator.to(self.device)

            # self.noise = torch.randn(1, self.nz, 1, 1, device=self.device)
            self.is_params_initialized = False


        def update_generator(self, batch_size=64, epoch=1000, log_interval=100):

            gen_loss_list = []
            best_loss = 100
            alpha = 0.1
            best_model_wts = copy.deepcopy(self.generator.state_dict())
            for i in range(1, epoch + 1):

                current_lr = self.adjust_learning_rate(i)
    
                self.generator.zero_grad()

                # noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
                noise = torch.randn(batch_size, self.nz, device=self.device)
                
                fake = self.generator(noise)
                
                output, feature = self.discriminator(fake)

                label = torch.full(
                    (batch_size,),
                    self.target_label,
                    dtype=torch.int64,
                    device=self.device,
                )

                loss_generator = self.generator_criterion(output, label)
                loss_activation = -feature.abs().mean() 

                loss_total = loss_generator +  alpha *loss_activation
                loss_total.backward()

                self.generator_optimizer.step()

                gen_loss = loss_total.item() 

                if log_interval != 0 and i % log_interval == 0:
                    gen_loss_list.append(gen_loss)
                    print(f"epoch {i}: total loss: {gen_loss} ;generator loss: {loss_generator} ;activation loss:{loss_activation.item()};lr:{current_lr}")


                    if gen_loss < best_loss:
                        best_loss = gen_loss
                        best_model_wts = copy.deepcopy(self.generator.state_dict())

            self.generator.load_state_dict(best_model_wts)
            # print("best_loss",best_loss)

            return loss_generator.item(), best_loss, self.generator

        def adjust_learning_rate(self, epoch):
            if epoch < 500:
                lr = 0.02
            elif epoch < 800:
                lr = 0.01
            else:
                lr = 0.005
            for param_group in self.generator_optimizer.param_groups:
                param_group['lr'] = lr
            return lr


        def update_discriminator(self):
            """Update the discriminator(global model)"""
            self.discriminator.load_state_dict(self.model.state_dict())

        def download(self, model_parameters):
            super().download(model_parameters)
            if ignore_first_download and not self.is_params_initialized:
                self.is_params_initialized = True
                return
            self.update_discriminator()
            self.update_generator(
                batch_size=gan_batch_size,
                epoch=gan_epoch,
                log_interval=gan_log_interval,
            )

        def attack(self, n):

            #noise = torch.randn(n, self.nz, 1, 1, device=self.device)
            noise = torch.randn(n, self.nz, device=self.device)
            with torch.no_grad():
                fake = self.generator(noise)
            return fake


    return GANAttackClientWrapper


class GANAttackManager():
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        return attach_ganattack_to_client(cls, *self.args, **self.kwargs)
