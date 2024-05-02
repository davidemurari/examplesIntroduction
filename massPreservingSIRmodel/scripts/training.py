from tqdm import tqdm

def train_network(model,trainloader,M,epochs,optimizer,criterion,scheduler):
  
  num_epochs = int(0.1 * epochs)

  for epoch in tqdm(range(epochs)):
      for i, data in enumerate(trainloader, 0):
          
          inputs, outputs = data[0], data[1]
          optimizer.zero_grad()
          loss = 0
          for k in range(M-1):
            inputs = model(inputs)
            loss += criterion(inputs,outputs[:,:,k])

          loss.backward()
          optimizer.step()
          
      if epoch % num_epochs == 0:  
          print('[Epoch %d] loss: %.10f' %
                (epoch + 1,loss.item()))

      scheduler.step()
      
  print('Finished Training')