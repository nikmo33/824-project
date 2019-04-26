from torch.utils.data import DataLoader
import torch
import numpy as np
from tensorboardX import SummaryWriter


class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10, simple=True):
        self.simple = simple
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 1000  # Steps

        self._train_dataset_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        # if self._cuda:
        self._model = self._model.cuda()

    # def _optimize(self, predicted_answers, true_answers):
    #     """
    #     This gets implemented in the subclasses. Don't implement this here.
    #     """
    #     raise NotImplementedError()

    def validate(self):
        for batch_id, batch_data in enumerate(self._val_dataset_loader):
            image_data = batch_data['image'].cuda()
            question_data = batch_data['question_onehot'].cuda()

            predicted_answer = self._model(
                image_data, question_data)
            ground_truth_answer = batch_data['answer_index'][0].cuda()
            # Optimize the model according to the predictions
            loss = self._optimize(
                predicted_answer, ground_truth_answer, train_flag=False)
            acc_val = self.accuracy(
                predicted_answer, ground_truth_answer, topk=(1,))
        return acc_val, loss

    def accuracy(self, output, target, topk=(1)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def train(self):
        writer = SummaryWriter()
        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                # ============
                # TODO: Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                # if self._cuda else batch_data['image']
                image_data = batch_data['image'].cuda()

                if (self.simple == True):
                    question_data = batch_data['question_onehot'].cuda()
                else:
                    question_data = batch_data['ques_embed'].cuda()

                predicted_answer = self._model(
                    image_data, question_data)
                predicted_answer = predicted_answer.cuda()

                ground_truth_answer = batch_data['answer_index'][0]
                ground_truth_answer = ground_truth_answer.cuda()

                # ============

                # Optimize the model according to the predictions
                loss = self._optimize(
                    predicted_answer, ground_truth_answer, train_flag=True)

                train_acc = self.accuracy(
                    predicted_answer, ground_truth_answer, topk=(1,))

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch,
                                                                      batch_id, num_batches, loss))
                    # TODO: you probably want to plot something here
                    writer.add_scalar('train/loss', loss, current_step)
                    writer.add_scalar('train/accuracy',
                                      train_acc[0][0], current_step)

                if current_step % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy, val_loss = self.validate()
                    print("Epoch: {} has val accuracy {}".format(
                        epoch, val_accuracy[0][0]))
                    # TODO: you probably want to plot something here
                    writer.add_scalar('validation/loss',
                                      val_loss, current_step)
                    writer.add_scalar('validation/accuracy',
                                      val_accuracy[0][0], current_step)
