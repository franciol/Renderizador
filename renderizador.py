# Desenvolvido por: Luciano Soares <lpsoares@insper.edu.br>
# Disciplina: Computação Gráfica
# Data: 28 de Agosto de 2020

import argparse     # Para tratar os parâmetros da linha de comando
import x3d          # Faz a leitura do arquivo X3D, gera o grafo de cena e faz traversal
import interface    # Janela de visualização baseada no Matplotlib
import gpu          # Simula os recursos de uma GPU

import numpy as np


PILHA_TRANSFORM = [np.identity(4)]

LARGURA = 900
ALTURA = 900


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
    # Taxa de Supersampling
    supersample = 3

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


def triangleSet(point, color):
    """ Função usada para renderizar TriangleSet. """
    # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
    # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
    # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
    # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
    # assim por diante.
    # No TriangleSet os triângulos são informados individualmente, assim os três
    # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
    # triângulo, e assim por diante.

    # O print abaixo é só para vocês verificarem o funcionamento, deve ser removido.
    # imprime no terminal pontos

    for i in range(0, len(point), 9):
        tmp_v1 = np.array([point[i], point[i+1], point[i+2], 1]).T
        tmp_v2 = np.array([point[i+3], point[i+4], point[i+5], 1]).T
        tmp_v3 = np.array([point[i+6], point[i+7], point[i+8], 1]).T

        v1 = np.dot(PILHA_TRANSFORM[-1], tmp_v1)
        v2 = np.dot(PILHA_TRANSFORM[-1], tmp_v2)
        v3 = np.dot(PILHA_TRANSFORM[-1], tmp_v3)

        triang = np.array([v1/np.sqrt(np.sum(v1**2)), v2/np.sqrt(np.sum(v2**2)), v3/np.sqrt(np.sum(v3**2))]).T


        conf_tela = np.dot(np.array([[LARGURA/2, 0, 0, LARGURA/2], [0, -ALTURA/2, 0, ALTURA/2], [0, 0, 1, 0],[0, 0, 0, 1]]) , triang).T

        points = []
        points.extend(conf_tela[0][:2])
        points.extend(conf_tela[1][:2])
        points.extend(conf_tela[2][:2])
        
        triangleSet2D(points, color)


def viewpoint(position, orientation, fieldOfView):
    """ Função usada para renderizar (na verdade coletar os dados) de Viewpoint. """
    # Na função de viewpoint você receberá a posição, orientação e campo de visão da
    # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
    # perspectiva para poder aplicar nos pontos dos objetos geométricos.

    if(orientation[0]):
        or_mat = np.array(
            [[1, 0, 0, 0], [0, np.cos(orientation[-1]), -np.sin(orientation[-1]), 0], [0, np.sin(orientation[-1]), np.cos(orientation[-1]), 0], [0, 0, 0, 1]])
    elif(orientation[1]):
        or_mat = np.array(
            [[np.cos(orientation[-1]), 0, np.sin(orientation[-1]), 0], [0, 1, 0, 0], [-np.sin(orientation[-1]), 0, np.cos(orientation[-1]), 0], [0, 0, 0, 1]])
    else:
        or_mat = np.array(
            [[np.cos(orientation[-1]), -np.sin(orientation[-1]), 0, 0], [np.sin(orientation[-1]), np.cos(orientation[-1]), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    look_at = np.dot(or_mat, np.array(
        [[1, 0, 0, -position[0]], [0, 1, 0, -position[1]], [0, 0, 1, -position[2]], [0, 0, 0, 1]]))
    PILHA_TRANSFORM.append(np.dot(look_at, PILHA_TRANSFORM[-1]))

    aspect = LARGURA/ALTURA
    near = 0.5
    top = near * np.tan(fieldOfView)
    far = 200
    right = top * aspect

    perspective = np.array([[near/right, 0, 0, 0], [0, near/top, 0, 0], [
        0, 0, -(far+near)/(far-near), -(2*far*near)/(far-near)], [0, 0, -1, 0]])

    PILHA_TRANSFORM.append(np.dot(perspective, PILHA_TRANSFORM[-1]))


def transform(translation, scale, rotation):
    """ Função usada para renderizar (na verdade coletar os dados) de Transform. """
    # A função transform será chamada quando se entrar em um nó X3D do tipo Transform
    # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
    # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
    # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
    # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
    # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
    # modelos do mundo em alguma estrutura de pilha.

    # O print abaixo é só para vocês verificarem o funcionamento, deve ser removido.
    if rotation[0]:
        rotat = np.array([[1, 0, 0, 0],
                          [0, np.cos(rotation[3]), -np.sin(rotation[3]), 0],
                          [0, np.sin(rotation[3]), np.cos(rotation[3]), 0],
                          [0, 0, 0, 1]])
    elif rotation[1]:
        rotat = np.array([[np.cos(rotation[3]), 0, np.sin(rotation[3]), 0],
                          [0, 1, 0, 0],
                          [-np.sin(rotation[3]), 0, np.cos(rotation[3]), 0],
                          [0, 0, 0, 1]])
    else:
        rotat = np.array([[np.cos(rotation[3]), -np.sin(rotation[3]), 0, 0],
                          [np.sin(rotation[3]), np.cos(rotation[3]), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

    transla = np.array([[1, 0, 0, translation[0]],
                        [0, 1, 0, translation[1]],
                        [0, 0, 1, translation[2]],
                        [0, 0, 0, 1]])
    scala = np.array([[scale[0], 0, 0, 0],
                      [0, scale[1], 0, 0],
                      [0, 0, scale[2], 0],
                      [0, 0, 0, 1]])

    PILHA_TRANSFORM.append(
        np.dot(PILHA_TRANSFORM[-1], np.dot(np.dot(transla, scala), rotat)))


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

    # O print abaixo é só para vocês verificarem o funcionamento, deve ser removido.
    triangs = []
    for i in range(int(stripCount[0])-2):
        cur = i
        triangs.extend([ point[cur*3] , point[3*cur+1] , point[3*cur+2]])
        cur = i+1
        triangs.extend([ point[cur*3] , point[3*cur+1] , point[3*cur+2]])
        cur = i+2
        triangs.extend([ point[cur*3] , point[3*cur+1] , point[3*cur+2]])
      
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
    triangs = []
    for i in range(len(index)-3):
        cur = int(index[int(i)])
        triangs.extend([ point[cur*3] , point[3*cur+1] , point[3*cur+2]])
        cur = int(index[int(i)+1])
        triangs.extend([ point[cur*3] , point[3*cur+1] , point[3*cur+2]])
        cur = int(index[int(i)+2])
        triangs.extend([ point[cur*3] , point[3*cur+1] , point[3*cur+2]])
      
    triangleSet(triangs, color)

def box(size, color):
    """ Função usada para renderizar Boxes. """
    # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
    # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
    # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
    # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
    # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
    # encontre os vértices e defina os triângulos.

    # O print abaixo é só para vocês verificarem o funcionamento, deve ser removido.
    print("Box : size = {0}".format(size))  # imprime no terminal pontos

    x = size[0]/2
    y = size[1]/2
    z = size[2]/2

    triangs = []

    # z positivo fixo
    triangs.extend([-x , y , z ])
    triangs.extend([-x ,-y , z ])
    triangs.extend([ x , y , z ])
    
    triangs.extend([ x ,-y , z ])
    triangs.extend([-x ,-y , z ])
    triangs.extend([ x , y , z ])

    # z negativo fixo
    triangs.extend([-x , y ,-z ])
    triangs.extend([-x ,-y ,-z ])
    triangs.extend([ x , y ,-z ])
    
    triangs.extend([ x ,-y ,-z ])
    triangs.extend([-x ,-y ,-z ])
    triangs.extend([ x , y ,-z ])

    # y positivo fixo
    triangs.extend([-x , y , z ])
    triangs.extend([-x , y ,-z ])
    triangs.extend([ x , y , z ])
    
    triangs.extend([ x , y , z ])
    triangs.extend([-x , y ,-z ])
    triangs.extend([ x , y ,-z ])

    # y negativo fixo
    triangs.extend([-x ,-y , z ])
    triangs.extend([-x ,-y ,-z ])
    triangs.extend([ x ,-y , z ])
    
    triangs.extend([ x ,-y , z ])
    triangs.extend([-x ,-y ,-z ])
    triangs.extend([ x ,-y ,-z ])

    # x positivo fixo
    triangs.extend([ x , y , z ])
    triangs.extend([ x ,-y ,-z ])
    triangs.extend([ x , y ,-z ])
    
    triangs.extend([ x , y , z ])
    triangs.extend([ x ,-y ,-z ])
    triangs.extend([ x ,-y , z ])

    # x negativo fixo
    triangs.extend([-x , y , z ])
    triangs.extend([-x ,-y ,-z ])
    triangs.extend([-x , y ,-z ])
    
    triangs.extend([-x , y , z ])
    triangs.extend([-x ,-y ,-z ])
    triangs.extend([-x ,-y , z ])

    triangleSet(triangs, color)
    
    
    


if __name__ == '__main__':

    # Valores padrão da aplicação
    width = LARGURA
    height = ALTURA
    x3d_file = "exemplo4.x3d"
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
