import matplotlib.pyplot as plt
import numpy as np
import pywt
import cv2 


img = cv2.imread('./cachorro.jpg', 0)  # Read as grayscale

# Convert pywt.dwt2
img_array = np.asarray(img)

# Perform the 2D DWT
coeffs2 = pywt.dwt2(img_array, 'db1')

# Extract coefficients
LL, (LH, HL, HH) = coeffs2

# Plot approximation and details
titles = ['Original','Approximation', 'Horizontal detail', 'Vertical detail', 'Diagonal detail']
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([img,LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 5, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()


# Aplique a transformada Wavelet (usando 'haar') para decompor a imagem em componentes: (LL, LH, HL, HH)
# Descarte os componentes de detalhes  (LH, HL, HH) e mantenha apenas o componente de baixa frequência (LL)
# Realize a reconstrução inversa usando apenas o componente LL e visualize a imagem reconstruída.
# Desafio: Compare a imagem original com a imagem reconstruída. Avalie a qualidade da imagem comprimida

# Carregar a imagem em escala de cinza
img = cv2.imread('./cachorro.jpg', 0)  # Lendo como escala de cinza
img_array = np.asarray(img)

# Aplicar a Transformada Wavelet 2D
coeffs2 = pywt.dwt2(img_array, 'haar')  # Usando 'haar' como exemplo
LL, (LH, HL, HH) = coeffs2

# Descartar os componentes de detalhes ao substituir LH, HL e HH por zeros
imagem_comprimida = (LL, (np.zeros_like(LH), np.zeros_like(HL), np.zeros_like(HH)))

# Realizar a reconstrução inversa usando apenas o componente LL
imagem_reconstruida = pywt.idwt2(imagem_comprimida, 'haar')

# Ajustar o tamanho da imagem reconstruída para que coincida com a original, se necessário
if imagem_reconstruida.shape != img.shape:
    imagem_reconstruida = cv2.resize(imagem_reconstruida, (img.shape[1], img.shape[0]))

# Normalizar a imagem reconstruída para o intervalo [0, 255] e converter para uint8
imagem_reconstruida = np.clip(imagem_reconstruida, 0, 255).astype(np.uint8)

# Exibir as imagens para comparação
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Imagem Original')
ax[0].axis('off')

ax[1].imshow(imagem_reconstruida, cmap='gray')
ax[1].set_title('Imagem Reconstruída (Apenas LL)')
ax[1].axis('off')

plt.show()

# Calcular o MSE (Erro Quadrático Médio) entre a imagem original e a reconstruída
mse = np.mean((img - imagem_reconstruida) ** 2)
print(f'Erro Quadrático Médio (MSE): {mse}')