# Desenvolvido por: Luciano Soares <lpsoares@insper.edu.br>
# Disciplina: Computação Gráfica
# Data: 28 de Agosto de 2020

import argparse     # Para tratar os parâmetros da linha de comando
import x3d          # Faz a leitura do arquivo X3D, gera o grafo de cena e faz traversal
import interface    # Janela de visualização baseada no Matplotlib
import gpu          # Simula os recursos de uma GPU

import numpy as np

# PILHA de transformações
PILHA_TRANSFORM = [np.identity(4)]

# Dimensoes tela
LARGURA = 1200
ALTURA = 720

# Taxa de Supersampling
supersample = 1


def polypoint2D(point, color):
    """ Função usada para renderizar Polypoint2D. """
    # Converte o esquema de cores
    color = [int(i * 255) for i in color]
    for i in range(int(len(point)//2)):
        # Transforma de Float para inteiro cada X e Y
        y = int(point[(i*2)])
        x = int(point[(2*i)+1])

        # Define onde desenhar o pixel e com que cor
        gpu.GPU.set_pixel(
            y, x, color[0], color[1], color[2])


def polyline2D(lineSegments, color):
    """ Função usada para renderizar Polyline2D. """
    y1 = (lineSegments[0])
    y2 = (lineSegments[2])
    x1 = (lineSegments[1])
    x2 = (lineSegments[3])
    dx = abs(x2-x1)
    dy = abs(y2-y1)

    # Decide se usa X ou Y como base do FOR
    if(dx > dy):

        if(x1 < x2):
            # Comeca no ponto 1 e vai para ponto 2 - FOR em X
            s = (y2-y1)/(x2-x1)
            v = y1
            for u in range(int(x1), int(x2)+1):
                # altera um pixel da imagem
                gpu.GPU.set_pixel(int((v)//1), u, 255 *
                                  color[0], 255*color[1], 255*color[2])
                v += s
        else:
            # Comeca no ponto 2 e vai para ponto 1 - FOR em X
            s = (y1-y2)/(x1-x2)
            v = y2
            for u in range(int(x2), int(x1)+1):
                # altera um pixel da imagem
                gpu.GPU.set_pixel(int((v)//1), u, 255 *
                                  color[0], 255*color[1], 255*color[2])
                v += s

    else:

        if(y1 < y2):
            # Comeca no ponto 1 e vai para ponto 2 - FOR em Y
            s = (x2-x1)/(y2-y1)
            u = x1
            for v in range(int(y1), int(y2)+1):
                # altera um pixel da imagem
                gpu.GPU.set_pixel(v, int((u)//1), 255 *
                                  color[0], 255*color[1], 255*color[2])
                u += s
        else:
            # Comeca no ponto 2 e vai para ponto 1 - FOR em Y
            s = (x1-x2)/(y1-y2)
            u = x2
            for v in range(int(y2), int(y1)+1):
                # altera um pixel da imagem
                gpu.GPU.set_pixel(v, int((u)//1), 255 *
                                  color[0], 255*color[1], 255*color[2])
                u += s

        # gpu.GPU.set_pixel( int((v)//1), u, 255*color[0], 255*color[1], 255*color[2]) # altera um pixel da imagem


def triangleSet2D(vertices, color):
    """ Função usada para renderizar TriangleSet2D. """

    # Salva os valores originais dos vertices
    y0_orig = vertices[0]
    x0_orig = vertices[1]
    y1_orig = vertices[2]
    x1_orig = vertices[3]
    y2_orig = vertices[4]
    x2_orig = vertices[5]

    # Valores dos vertices com supersampling
    y0 = y0_orig * supersample
    x0 = x0_orig * supersample
    y1 = y1_orig * supersample
    x1 = x1_orig * supersample
    y2 = y2_orig * supersample
    x2 = x2_orig * supersample

    x_min = round(min(x0, x1, x2))
    x_max = round(max(x0, x1, x2))
    y_min = round(min(y0, y1, y2))
    y_max = round(max(y0, y1, y2))

    # lista que vai guardar as coordenadas dos pixeis a serem desenhados, com supersampling
    lista_miniPixels = []

    # Verifica para cada pixel dentro do quadrado que contem o triangulo, se deve desenha-lo
    # calc0,calc1,calc2
    for v in range(y_min-1, y_max+1):
        for u in range(x_min-1, x_max+1):

            calc1 = (y1-y2)*(u+0.5) + (x2-x1)*(v+0.5) + (x1*y2 - x2*y1)
            if((y1-y2)*(x0) + (x2-x1)*(y0) + (x1*y2 - x2*y1) < 0):
                calc1 *= -1

            if(calc1 >= 0):
                calc2 = (y0-y2)*(u+0.5) + (x2-x0)*(v+0.5) + (x0*y2 - x2*y0)
                if((y0-y2)*(x1) + (x2-x0)*(y1) + (x0*y2 - x2*y0) < 0):
                    calc2 *= -1
                if(calc2 > 0):

                    calc3 = (y1-y0)*(u+0.5) + (x0-x1)*(v+0.5) + (x1*y0 - x0*y1)
                    if((y1-y0)*(x2) + (x0-x1)*(y2) + (x1*y0 - x0*y1) < 0):
                        calc3 *= -1
                    if(calc3 >= 0):
                        lista_miniPixels.append([u, v])

    x_min = round(min(x0_orig, x1_orig, x2_orig))
    x_max = round(max(x0_orig, x1_orig, x2_orig))
    y_min = round(min(y0_orig, y1_orig, y2_orig))
    y_max = round(max(y0_orig, y1_orig, y2_orig))

    for v in range(y_min-1, y_max+1):
        for u in range(x_min-1, x_max+1):
            intensity = sum(1 for i in lista_miniPixels if (
                i[0]//supersample == u and i[1]//supersample == v))/(supersample**2)
            # altera um pixel da imagem

            if(intensity > 0):
                gpu.GPU.set_pixel(
                    v, u, 255*color[0]*intensity, 255*color[1]*intensity, 255*color[2]*intensity)

def triangleSet2DTexture(vertices, coords, indx, matrix):
    """ Função usada para renderizar TriangleSet2D. """
    matrix = gpu.GPU.load_texture(matrix[0])
    # Salva os valores originais dos vertices

    y0_orig = vertices[0]
    x0_orig = vertices[1]
    y1_orig = vertices[2]
    x1_orig = vertices[3]
    y2_orig = vertices[4]
    x2_orig = vertices[5]

    # Valores dos vertices com supersampling
    y0 = y0_orig * supersample
    x0 = x0_orig * supersample
    y1 = y1_orig * supersample
    x1 = x1_orig * supersample
    y2 = y2_orig * supersample
    x2 = x2_orig * supersample

    x_min = round(min(x0, x1, x2))
    x_max = round(max(x0, x1, x2))
    y_min = round(min(y0, y1, y2))
    y_max = round(max(y0, y1, y2))

    # lista que vai guardar as coordenadas dos pixeis a serem desenhados, com supersampling
    lista_miniPixels = []

    # Verifica para cada pixel dentro do quadrado que contem o triangulo, se deve desenha-lo
    # calc0,calc1,calc2
    y_l = range(y_min-1, y_max+1)
    x_l = range(x_min-1, x_max+1)
    for idx_v, v in enumerate(y_l):
        for idx_u, u in enumerate(x_l):


            calc1 = (y1-y2)*(u+0.5) + (x2-x1)*(v+0.5) + (x1*y2 - x2*y1)
            if((y1-y2)*(x0) + (x2-x1)*(y0) + (x1*y2 - x2*y1) < 0):
                calc1 *= -1

            if(calc1 >= 0):
                calc2 = (y0-y2)*(u+0.5) + (x2-x0)*(v+0.5) + (x0*y2 - x2*y0)
                if((y0-y2)*(x1) + (x2-x0)*(y1) + (x0*y2 - x2*y0) < 0):
                    calc2 *= -1
                if(calc2 > 0):

                    calc3 = (y1-y0)*(u+0.5) + (x0-x1)*(v+0.5) + (x1*y0 - x0*y1)
                    if((y1-y0)*(x2) + (x0-x1)*(y2) + (x1*y0 - x0*y1) < 0):
                        calc3 *= -1
                    if(calc3 >= 0):
                        
                        lista_miniPixels.append([u, v])
                        

    x_min = round(min(x0_orig, x1_orig, x2_orig))
    x_max = round(max(x0_orig, x1_orig, x2_orig))
    y_min = round(min(y0_orig, y1_orig, y2_orig))
    y_max = round(max(y0_orig, y1_orig, y2_orig))

    y_l = range(y_min-1, y_max+1)
    x_l = range(x_min-1, x_max+1)
    for idx_v, v in enumerate(y_l):
        for idx_u, u in enumerate(x_l):


            intensity = sum(1 for i in lista_miniPixels if (
                i[0]//supersample == u and i[1]//supersample == v))/(supersample**2)
            # altera um pixel da imagem

            if(intensity > 0):
                idx_u1 = int(idx_u/len(x_l)*matrix.shape[1])
                idx_v1 = int(idx_v/len(y_l)*matrix.shape[0])


                color = matrix[idx_u1,idx_v1]


                gpu.GPU.set_pixel(
                    v, u, color[0]*intensity, color[1]*intensity, color[2]*intensity)

def triangleSet2DColor(vertices, colors):
    """ Função usada para renderizar TriangleSet2D. """

    # Salva os valores originais dos vertices
    y0_orig = vertices[0]
    x0_orig = vertices[1]
    y1_orig = vertices[2]
    x1_orig = vertices[3]
    y2_orig = vertices[4]
    x2_orig = vertices[5]

    c1 = colors[0:3]
    c2 = colors[3:6]
    c3 = colors[6:9]

    # Valores dos vertices com supersampling
    y0 = y0_orig * supersample
    x0 = x0_orig * supersample
    y1 = y1_orig * supersample
    x1 = x1_orig * supersample
    y2 = y2_orig * supersample
    x2 = x2_orig * supersample

    x_min = round(min(x0, x1, x2))
    x_max = round(max(x0, x1, x2))
    y_min = round(min(y0, y1, y2))
    y_max = round(max(y0, y1, y2))

    # lista que vai guardar as coordenadas dos pixeis a serem desenhados, com supersampling
    lista_miniPixels = []

    # Verifica para cada pixel dentro do quadrado que contem o triangulo, se deve desenha-lo
    # calc0,calc1,calc2
    for v in range(y_min-1, y_max+1):
        for u in range(x_min-1, x_max+1):

            calc1 = (y1-y2)*(u+0.5) + (x2-x1)*(v+0.5) + (x1*y2 - x2*y1)
            if((y1-y2)*(x0) + (x2-x1)*(y0) + (x1*y2 - x2*y1) < 0):
                calc1 *= -1

            if(calc1 >= 0):
                calc2 = (y0-y2)*(u+0.5) + (x2-x0)*(v+0.5) + (x0*y2 - x2*y0)
                if((y0-y2)*(x1) + (x2-x0)*(y1) + (x0*y2 - x2*y0) < 0):
                    calc2 *= -1
                if(calc2 > 0):

                    calc3 = (y1-y0)*(u+0.5) + (x0-x1)*(v+0.5) + (x1*y0 - x0*y1)
                    if((y1-y0)*(x2) + (x0-x1)*(y2) + (x1*y0 - x0*y1) < 0):
                        calc3 *= -1
                    if(calc3 >= 0):
                        lista_miniPixels.append([u, v])

    x_min = round(min(x0_orig, x1_orig, x2_orig))
    x_max = round(max(x0_orig, x1_orig, x2_orig))
    y_min = round(min(y0_orig, y1_orig, y2_orig))
    y_max = round(max(y0_orig, y1_orig, y2_orig))

    for v in range(y_min-1, y_max+1):
        for u in range(x_min-1, x_max+1):
            intensity = sum(1 for i in lista_miniPixels if (
                i[0]//supersample == u and i[1]//supersample == v))/(supersample**2)

            # altera um pixel da imagem
            alpha = -(u-x1)*(y2-y1)+(v-y1)*(x2-x1) / \
                (-(x0-x1)*(y2-y1)+(y0-y1)*(x2-x1))

            beta = -(u-x2)*(y0-y2)+(v-y2)*(x0-x2) / \
                (-(x1-x2)*(y0-y2)+(y1-y2)*(x0-x2))

            gama = 1 - alpha - beta
            
            color_pix = np.add(np.multiply(c1,alpha),np.multiply(c2,beta))
            
            color_pix = np.add(color_pix,np.multiply(c3,gama))
            
            color_pix = np.multiply(color_pix,(255*intensity))
            
            if(intensity > 0):
                gpu.GPU.set_pixel(
                    v, u, color_pix[0], color_pix[1], color_pix[2])


def triangleSet(point, color, colors=False, Textures=False, current_texture=None, text_coords=None, text_color_index=None):
    """ Função usada para renderizar TriangleSet. """
    # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
    # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
    # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
    # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
    # assim por diante.
    # No TriangleSet os triângulos são informados individualmente, assim os três
    # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
    # triângulo, e assim por diante.

    # For itera por cada triangulo
    for i in range(0, len(point), 9):
        tmp_v1 = np.array([point[i], point[i+1], point[i+2], 1]).T
        tmp_v2 = np.array([point[i+3], point[i+4], point[i+5], 1]).T
        tmp_v3 = np.array([point[i+6], point[i+7], point[i+8], 1]).T

        # Pontos dos triangulos ja posicionados em relacao á ultima transformação
        v1 = np.matmul(PILHA_TRANSFORM[-1], tmp_v1)
        v2 = np.matmul(PILHA_TRANSFORM[-1], tmp_v2)
        v3 = np.matmul(PILHA_TRANSFORM[-1], tmp_v3)

        # Matriz do triangulo já normalizada
        triang = np.array([v1/np.sqrt(np.sum(v1**2)), v2 /
                           np.sqrt(np.sum(v2**2)), v3/np.sqrt(np.sum(v3**2))]).T

        # transformação para coordenadas da tela
        conf_tela = np.matmul(np.array(
            [[LARGURA/2, 0, 0, LARGURA/2], [0, -ALTURA/2, 0, ALTURA/2], [0, 0, 1, 0], [0, 0, 0, 1]]), triang).T

        # Geração dos pontos 2D a desenhar
        points = []
        points.extend(conf_tela[0][:2])
        points.extend(conf_tela[1][:2])
        points.extend(conf_tela[2][:2])

        # Envio para a função de Desenho
        if(Textures):
            triangleSet2DTexture(points, text_coords, text_color_index, current_texture)
        elif(colors):
            triangleSet2DColor(points, color)
        else:
            triangleSet2D(points,color)

def viewpoint(position, orientation, fieldOfView):
    """ Função usada para renderizar (na verdade coletar os dados) de Viewpoint. """
    # Na função de viewpoint você receberá a posição, orientação e campo de visão da
    # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
    # perspectiva para poder aplicar nos pontos dos objetos geométricos.
    # Verifica que a orientação
    if(orientation[0]):
        or_mat = np.array(
            [[1, 0, 0, 0], [0, np.cos(orientation[-1]), -np.sin(orientation[-1]), 0], [0, np.sin(orientation[-1]), np.cos(orientation[-1]), 0], [0, 0, 0, 1]])
    elif(orientation[1]):
        or_mat = np.array(
            [[np.cos(orientation[-1]), 0, np.sin(orientation[-1]), 0], [0, 1, 0, 0], [-np.sin(orientation[-1]), 0, np.cos(orientation[-1]), 0], [0, 0, 0, 1]])
    else:
        or_mat = np.array(
            [[np.cos(orientation[-1]), -np.sin(orientation[-1]), 0, 0], [np.sin(orientation[-1]), np.cos(orientation[-1]), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Matriz Look At
    look_at = np.matmul(or_mat, np.array(
        [[1, 0, 0, -position[0]], [0, 1, 0, -position[1]], [0, 0, 1, -position[2]], [0, 0, 0, 1]]))

    # Adição da transformação á pilha
    PILHA_TRANSFORM.append(np.matmul(look_at, PILHA_TRANSFORM[-1]))

    # Matriz de Perspectiva
    aspect = LARGURA/ALTURA
    near = 0.5
    top = near * np.tan(fieldOfView)
    far = 200
    right = top * aspect

    perspective = np.array([[near/right, 0, 0, 0], [0, near/top, 0, 0], [
        0, 0, -(far+near)/(far-near), -(2*far*near)/(far-near)], [0, 0, -1, 0]])

    # Adição da transformação á pilha
    PILHA_TRANSFORM.append(np.matmul(perspective, PILHA_TRANSFORM[-1]))


def transform(translation, scale, rotation):
    """ Função usada para renderizar (na verdade coletar os dados) de Transform. """
    # A função transform será chamada quando se entrar em um nó X3D do tipo Transform
    # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
    # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
    # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
    # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
    # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
    # modelos do mundo em alguma estrutura de pilha.

    # Matriz de rotação
    if rotation[0]:
        rotat = np.array([[1, 0, 0, 0], [0, np.cos(rotation[3]), -np.sin(rotation[3]), 0],
                          [0, np.sin(rotation[3]), np.cos(rotation[3]), 0], [0, 0, 0, 1]])
    elif rotation[1]:
        rotat = np.array([[np.cos(rotation[3]), 0, np.sin(rotation[3]), 0], [
                         0, 1, 0, 0], [-np.sin(rotation[3]), 0, np.cos(rotation[3]), 0], [0, 0, 0, 1]])
    else:
        rotat = np.array([[np.cos(rotation[3]), -np.sin(rotation[3]), 0, 0], [np.sin(
            rotation[3]), np.cos(rotation[3]), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Matriz de translação
    transla = np.array([[1, 0, 0, translation[0]], [0, 1, 0, translation[1]], [
                       0, 0, 1, translation[2]], [0, 0, 0, 1]])

    # Matriz de Escala
    scala = np.array([[scale[0], 0, 0, 0], [0, scale[1], 0, 0], [
                     0, 0, scale[2], 0], [0, 0, 0, 1]])

    # Adição da transformação á pilha
    PILHA_TRANSFORM.append(
        np.matmul(PILHA_TRANSFORM[-1], np.matmul(np.matmul(transla, scala), rotat)))


def _transform():
    """ Função usada para renderizar (na verdade coletar os dados) de Transform. """
    # A função _transform será chamada quando se sair em um nó X3D do tipo Transform do
    # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
    # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
    # pilha implementada.
    PILHA_TRANSFORM.pop()


def triangleStripSet(point, stripCount, color):
    """ Função usada para renderizar TriangleStripSet. """
    # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
    # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
    # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
    # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
    # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
    # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
    # em uma lista chamada stripCount (perceba que é uma lista).

    # Cria a lista de pontos e adiciona todos os pontos da lista
    triangs = []
    for i in range(int(stripCount[0])-2):
        cur = i
        triangs.extend([point[cur*3], point[3*cur+1], point[3*cur+2]])
        cur = i+1
        triangs.extend([point[cur*3], point[3*cur+1], point[3*cur+2]])
        cur = i+2
        triangs.extend([point[cur*3], point[3*cur+1], point[3*cur+2]])

    triangleSet(triangs, color)


def indexedTriangleStripSet(point, index, color):
    """ Função usada para renderizar IndexedTriangleStripSet. """
    # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
    # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
    # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
    # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
    # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
    # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
    # como conectar os vértices é informada em index, o valor -1 indica que a lista
    # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
    # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
    # depois 2, 3 e 4, e assim por diante.

    # Cria a lista de pontos e adiciona todos os pontos da lista
    triangs = []
    for i in range(len(index)-3):
        cur = int(index[int(i)])
        triangs.extend([point[cur*3], point[3*cur+1], point[3*cur+2]])
        cur = int(index[int(i)+1])
        triangs.extend([point[cur*3], point[3*cur+1], point[3*cur+2]])
        cur = int(index[int(i)+2])
        triangs.extend([point[cur*3], point[3*cur+1], point[3*cur+2]])

    triangleSet(triangs, color)


def box(size, color):
    """ Função usada para renderizar Boxes. """
    # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
    # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
    # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
    # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
    # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
    # encontre os vértices e defina os triângulos.
    # coordenadas positivas da caixa
    x = size[0]/2
    y = size[1]/2
    z = size[2]/2
    # lista de pontos
    triangs = []
    # z negativo fixo
    triangs.extend([-x, y, -z])
    triangs.extend([-x, -y, -z])
    triangs.extend([x, y, -z])
    triangs.extend([x, -y, -z])
    triangs.extend([-x, -y, -z])
    triangs.extend([x, y, -z])
    # y positivo fixo
    triangs.extend([-x, y, z])
    triangs.extend([-x, y, -z])
    triangs.extend([x, y, z])
    triangs.extend([x, y, z])
    triangs.extend([-x, y, -z])
    triangs.extend([x, y, -z])
    # y negativo fixo
    triangs.extend([-x, -y, z])
    triangs.extend([-x, -y, -z])
    triangs.extend([x, -y, z])
    triangs.extend([x, -y, z])
    triangs.extend([-x, -y, -z])
    triangs.extend([x, -y, -z])
    # x positivo fixo
    triangs.extend([x, y, z])
    triangs.extend([x, -y, -z])
    triangs.extend([x, y, -z])
    triangs.extend([x, y, z])
    triangs.extend([x, -y, -z])
    triangs.extend([x, -y, z])
    # x negativo fixo
    triangs.extend([-x, y, z])
    triangs.extend([-x, -y, -z])
    triangs.extend([-x, y, -z])
    triangs.extend([-x, y, z])
    triangs.extend([-x, -y, -z])
    triangs.extend([-x, -y, z])
    # z positivo fixo
    triangs.extend([-x, y, z])
    triangs.extend([-x, -y, z])
    triangs.extend([x, y, z])
    triangs.extend([x, -y, z])
    triangs.extend([-x, -y, z])
    triangs.extend([x, y, z])
    # manda os pontos para desenhar
    triangleSet(triangs, color)


def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex, texCoord, texCoordIndex, current_color, current_texture):
    """ Função usada para renderizar IndexedFaceSet. """
    # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
    # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
    # Você receberá as coordenadas dos pontos no parâmetro cord, esses
    # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
    # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
    # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
    # segundo ponto e assim por diante. No IndexedFaceSet uma lista informando
    # como conectar os vértices é informada em coordIndex, o valor -1 indica que a lista
    # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
    # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
    # depois 2, 3 e 4, e assim por diante.
    # Adicionalmente essa implementação do IndexedFace suporta cores por vértices, assim
    # a se a flag colorPerVertex estiver habilidades, os vértices também possuirão cores
    # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
    # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
    # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
    # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
    # implementadado um método para a leitura de imagens.
    if (colorPerVertex):
        for i in range(0, len(coordIndex)-2):
            if(not ((-1) == coordIndex[i] or (-1) == coordIndex[i+1] or coordIndex[i+2] == (-1))):
                p1 = coord[coordIndex[i]*3:(coordIndex[i]+1)*3]
                p2 = coord[coordIndex[i+1]*3:(coordIndex[i+1]+1)*3]
                p3 = coord[coordIndex[i+2]*3:(coordIndex[i+2]+1)*3]

                c1 = color[colorIndex[i]*3:(colorIndex[i]+1)*3]
                c2 = color[colorIndex[i+1]*3:(colorIndex[i+1]+1)*3]
                c3 = color[colorIndex[i+2]*3:(colorIndex[i+2]+1)*3]

                c1.extend(c2)
                c1.extend(c2)
                p1.extend(p2)
                p1.extend(p3)
                triangleSet(p1, c1, True)
                
    elif(texCoord):
        points_txt = 0
        for i in range(0, len(coordIndex)-2):
            if(not ((-1) == coordIndex[i] or (-1) == coordIndex[i+1] or coordIndex[i+2] == (-1))):
                p1 = coord[coordIndex[i]*3:(coordIndex[i]+1)*3]
                p2 = coord[coordIndex[i+1]*3:(coordIndex[i+1]+1)*3]
                p3 = coord[coordIndex[i+2]*3:(coordIndex[i+2]+1)*3]
                p1.extend(p2)
                p1.extend(p3)
                txt = []
                txt.extend([texCoordIndex[points_txt*3+points_txt],texCoordIndex[points_txt*3+1+points_txt],texCoordIndex[points_txt*3+2+points_txt]])
                points_txt+=1
                triangleSet(p1, None, False, True, current_texture, texCoord,txt)
                
        
        
    else:
        for i in range(0, len(coordIndex)-2):
            if(not ((-1) == coordIndex[i] or (-1) == coordIndex[i+1] or coordIndex[i+2] == (-1))):
                p1 = coord[coordIndex[i]*3:(coordIndex[i]+1)*3]
                p2 = coord[coordIndex[i+1]*3:(coordIndex[i+1]+1)*3]
                p3 = coord[coordIndex[i+2]*3:(coordIndex[i+2]+1)*3]
                p1.extend(p2)
                p1.extend(p3)
                triangleSet(p1, current_color, False)


if __name__ == '__main__':

    # Valores padrão da aplicação
    width = LARGURA
    height = ALTURA
    x3d_file = "exemplo9.x3d"
    image_file = "tela.png"

    # Tratando entrada de parâmetro
    # parser para linha de comando
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--input", help="arquivo X3D de entrada")
    parser.add_argument("-o", "--output", help="arquivo 2D de saída (imagem)")
    parser.add_argument("-w", "--width", help="resolução horizonta", type=int)
    parser.add_argument("-h", "--height", help="resolução vertical", type=int)
    parser.add_argument(
        "-q", "--quiet", help="não exibe janela de visualização", action='store_true')
    args = parser.parse_args()  # parse the arguments
    if args.input:
        x3d_file = args.input
    if args.output:
        image_file = args.output
    if args.width:
        width = args.width
    if args.height:
        height = args.height

    # Iniciando simulação de GPU
    gpu.GPU(width, height, image_file)

    # Abre arquivo X3D
    scene = x3d.X3D(x3d_file)
    scene.set_resolution(width, height)

    # funções que irão fazer o rendering
    x3d.X3D.render["Polypoint2D"] = polypoint2D
    x3d.X3D.render["Polyline2D"] = polyline2D
    x3d.X3D.render["TriangleSet2D"] = triangleSet2D
    x3d.X3D.render["TriangleSet"] = triangleSet
    x3d.X3D.render["Viewpoint"] = viewpoint
    x3d.X3D.render["Transform"] = transform
    x3d.X3D.render["_Transform"] = _transform
    x3d.X3D.render["TriangleStripSet"] = triangleStripSet
    x3d.X3D.render["IndexedTriangleStripSet"] = indexedTriangleStripSet
    x3d.X3D.render["Box"] = box
    x3d.X3D.render["IndexedFaceSet"] = indexedFaceSet

    # Se no modo silencioso não configurar janela de visualização
    if not args.quiet:
        window = interface.Interface(width, height)
        scene.set_preview(window)

    scene.parse()  # faz o traversal no grafo de cena

    # Se no modo silencioso salvar imagem e não mostrar janela de visualização
    if args.quiet:
        gpu.GPU.save_image()  # Salva imagem em arquivo
    else:
        window.image_saver = gpu.GPU.save_image  # pasa a função para salvar imagens
        window.preview(gpu.GPU._frame_buffer)  # mostra janela de visualização
