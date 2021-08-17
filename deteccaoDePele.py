import cv2 #importação da biblioteca cv2
from google.colab.patches import cv2_imshow #importação do patche para imagem ser plotada no Google Colab

imagem = cv2.imread('/content/mulher.jpg') #ler a imagem 
cascade_path = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml') #ler o arquivo modelo, ja treinado

imagem_convertida_cinza = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY) #converte a imagem original para cinza

detector = cascade_path.detectMultiScale(imagem_convertida_cinza,1.3,10)#aplica o modelo cascade na imagem cinza

len(detector) #percorre a imagem e aplica borda na detecção
for(x,y,largura,altura) in detector:
  cv2.rectangle(imagem,(x,y), (x+largura,y+altura), (255,255,0),2)
  
cv2_imshow(imagem) #plota a imagem com o border box