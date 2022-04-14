import argparse
import json

import PIL
import nltk
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torchtext.data.metrics import bleu_score
from PIL.Image import Image
from matplotlib import cm, image as mpimg

from FlickrDataLoder import get_loader
from FlickrJSON import FlickrJSON
from FlickrVocab import FlickrVocab
from model import EncoderCNN, DecoderRNN
import torch.nn.utils.rnn
from torchvision import transforms
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    # print("Hello",vocab.idx2word[0])
    # Build data loader
    data_loader,Validationdata_loader,Testdata_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    flickj = FlickrJSON()
    JSON_data = flickj.BuildJson(args.caption_path)
    pythonObj = json.loads(JSON_data)

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the models
    total_step = len(data_loader)
    valtotal_step = len(Validationdata_loader)

    trainLossList = []
    valLossList = []
    for epoch in range(args.num_epochs):
        epochtrainloss = []
        epochtestloss = []
        for i, (images, captions, lengths,file_name) in enumerate(data_loader):
            print("Train Batch:", i)
            # Set mini-batch dataset
            images = images.to(device)
            print("Images:", len(images), " captions:", len(captions))
            captions = captions.to(device)
            targets = torch.nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # print(torch.nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True))
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            # print(targets.shape)
            # print(outputs.shape)
            loss = criterion(outputs, targets)
            torch.set_printoptions(threshold=100000)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()
            epochtrainloss.append(loss.item())
            # print("outputs length:", len(outputs))
            # print("targets length: ", len(targets))
            # Print log info

            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
            top5 = accuracy(outputs, targets, 5)
            print("Train Accuracy", top5)

            # Save the model checkpoints
            if (i + 1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        for j, (images, captions, lengths,file_name) in enumerate(Validationdata_loader):
            print("Validation Batch:", j)
            # Set mini-batch dataset
            images = images.to(device)
            print("Images:", len(images), " captions:", len(captions))
            captions = captions.to(device)
            targets = torch.nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # print(torch.nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True))
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            valloss = criterion(outputs, targets)
            torch.set_printoptions(threshold=100000)
            epochtestloss.append(valloss.item())
            # print("outputs length:", len(outputs))
            # print("targets length: ", len(targets))
            # Print log info
            if j % 5 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Validation Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, j, valtotal_step, valloss.item(), np.exp(valloss.item())))
            top5 = accuracy(outputs, targets, 5)
            print("Validation Accuracy:", top5)

        valLossList.append(np.mean(epochtestloss))
        trainLossList.append(np.mean(epochtrainloss))

    for k, (images, captions, lengths,file_name) in enumerate(Testdata_loader):
        print("Testing Batch:", k)
        # Set mini-batch dataset
        cpuimages = images
        images = images.to(device)
        print("Images:", len(images), " captions:", len(captions))
        cpucaptions = captions.numpy()
        captions = captions.to(device)
        targets = torch.nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True)[0]
        # print(torch.nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True))
        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder.sample(features)
        cpuoutputs = outputs.cpu().detach().numpy()
        for f in range(10):
            print("image:", f,"  ",file_name[f])
            predictedcaption = []
            groundcaption = []
            lists = []
            for i in range(len(pythonObj['annotations'])):
                if(pythonObj['annotations'][i]['file_name']  == file_name[f]):
                        caption = str(pythonObj['annotations'][i]['caption'])
                        tokens = nltk.tokenize.word_tokenize(caption.lower())
                        lists.append(tokens)

            for k1 in range(len(cpuoutputs[f])):
                predictedcaption.append(vocab.idx2word[cpuoutputs[f][k1]])
            for k1 in range(len(cpucaptions[f])):
                groundcaption.append(vocab.idx2word[cpucaptions[f][k1]])
            print("Predicted:",' '.join(predictedcaption))
            print("All captions:",lists)


            for candidate in lists:
                print("Groundtruth:", ' '.join(candidate))
                score = sentence_bleu(predictedcaption, candidate)
                print("BLEU SCORE:",score)
            txt = ''
            xxx = (mpimg.imread("./Flickr8k_Dataset/Flicker_Resized/"+ file_name[f]))
            plt.title("")

            plt.title("Predicted: "+' '.join(predictedcaption), fontsize=8)
            plt.figtext(0.5, 0.01, "Ground Truth: " +' '.join(groundcaption),
                        wrap=True, horizontalalignment='center', fontsize=8)
            plt.axis('off')
            plt.imshow(xxx)

        if(k==0):
            break
    # reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
    # candidate = ['this', 'is', 'a', 'test']

    plt.plot(trainLossList)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.show()

    plt.plot(valLossList)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.show()

    plt.plot(trainLossList)
    plt.plot(valLossList)
    plt.xlabel("Epochs")
    plt.ylabel("Training vs Validation Loss")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./Flickr8k_text/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./Flickr8k_Dataset/Flicker_Resized', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='./Flickr8k_text/Flickr8k.token.txt',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)