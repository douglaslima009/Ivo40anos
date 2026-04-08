import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import trimesh
import pyrr
import sys
import random

# --- 1. SHADERS ---
PHONG_VERT = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
out vec3 FragPos; out vec3 Normal;
uniform mat4 model; uniform mat4 view; uniform mat4 projection;
void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

PHONG_FRAG = """
#version 330 core
out vec4 FragColor;
in vec3 Normal; in vec3 FragPos;
uniform vec3 viewPos; uniform vec3 objectColor;
void main() {
    float ambientStrength = 0.5; vec3 ambient = ambientStrength * vec3(1.0);
    vec3 norm = normalize(Normal); vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8));
    float diff = max(dot(norm, lightDir), 0.0); vec3 diffuse = diff * vec3(0.5);
    float specularStrength = 0.7; vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * vec3(1.0);
    vec3 result = (ambient + diffuse + specular) * objectColor;
    float distance = length(viewPos - FragPos);
    float fogFactor = clamp((distance - 5.0) / 20.0, 0.0, 1.0);
    vec3 finalColor = mix(result, vec3(1.0, 1.0, 1.0), fogFactor);
    FragColor = vec4(finalColor, 1.0);
}
"""

CROSSHAIR_VERT = "#version 330 core\nlayout (location = 0) in vec2 aPos;\nvoid main() { gl_Position = vec4(aPos, 0.0, 1.0); }"
CROSSHAIR_FRAG = "#version 330 core\nout vec4 FragColor;\nvoid main() { FragColor = vec4(0.0, 0.0, 0.0, 1.0); }"
UI_VERT = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;
out vec2 TexCoords;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); TexCoords = aTexCoords; }
"""
UI_FRAG = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoords;
uniform sampler2D textTexture;
void main() { FragColor = texture(textTexture, TexCoords); }
"""

# --- 2. FUNCOES DE SUPORTE ---
def carregar_modelo_completo(caminho_glb):
    cena = trimesh.load(caminho_glb)
    dumped_meshes = cena.dump() if isinstance(cena, trimesh.Scene) else [cena]

    partes = []
    for malha in dumped_meshes:
        cor_padrao = [0.5, 0.5, 0.5] 
        try:
            if hasattr(malha.visual, 'material') and hasattr(malha.visual.material, 'pbr_metallic_roughness'):
                 pbr = malha.visual.material.pbr_metallic_roughness
                 if pbr.base_color_factor is not None:
                     cor_padrao = [pbr.base_color_factor[0], pbr.base_color_factor[1], pbr.base_color_factor[2]]
            elif hasattr(malha.visual, 'kind') and malha.visual.kind == 'vertex':
                c_avg = np.mean(malha.visual.vertex_colors, axis=0) / 255.0
                cor_padrao = [c_avg[0], c_avg[1], c_avg[2]]
        except: pass

        faces = malha.faces.flatten()
        vertices = np.array(malha.vertices[faces], dtype=np.float32)
        normais = np.array(malha.vertex_normals[faces], dtype=np.float32)
        vertex_data = np.hstack([vertices, normais]).flatten()
        
        vao = glGenVertexArrays(1); vbo = glGenBuffers(1)
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        partes.append({'vao': vao, 'count': len(faces), 'color': cor_padrao})
    return partes

def criar_esfera_vao():
    malha = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    faces = malha.faces.flatten()
    vertices = np.array(malha.vertices[faces], dtype=np.float32)
    normais = np.array(malha.vertex_normals[faces], dtype=np.float32)
    vertex_data = np.hstack([vertices, normais]).flatten()
    vao = glGenVertexArrays(1); vbo = glGenBuffers(1)
    glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    return vao, len(faces)

def criar_crosshair_vao():
    cross_data = np.array([-0.02, 0.0, 0.02, 0.0, 0.0, -0.02, 0.0, 0.02], dtype=np.float32)
    vao = glGenVertexArrays(1); vbo = glGenBuffers(1)
    glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, cross_data.nbytes, cross_data, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    return vao, 4

def criar_ui_quad_vao():
    quad_data = np.array([-0.8, -0.9, 0.0, 0.0,  0.8, -0.9, 1.0, 0.0,  0.8, -0.5, 1.0, 1.0, -0.8, -0.5, 0.0, 1.0], dtype=np.float32)
    vao = glGenVertexArrays(1); vbo = glGenBuffers(1)
    glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, quad_data.nbytes, quad_data, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
    glEnableVertexAttribArray(1)
    return vao

def gerar_textura_texto(texto, fonte):
    surface = pygame.Surface((800, 150), pygame.SRCALPHA)
    surface.fill((0, 0, 0, 200)) 
    palavras = texto.split()
    linhas = []; linha_atual = ""
    for p in palavras:
        if fonte.size(linha_atual + p)[0] < 760: linha_atual += p + " "
        else: linhas.append(linha_atual); linha_atual = p + " "
    linhas.append(linha_atual)
    y_offset = 20
    for linha in linhas:
        surface.blit(fonte.render(linha, True, (255, 255, 255)), (20, y_offset))
        y_offset += 35
    text_data = pygame.image.tostring(surface, "RGBA", True)
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 800, 150, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
    return tex_id

# --- 3. APLICACAO PRINCIPAL ---
def main():
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init(); pygame.font.init(); pygame.mixer.init()
    largura, altura = 1024, 768
    tela = pygame.display.set_mode((largura, altura), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("IVO 40 ANOS")
    pygame.event.set_grab(True); pygame.mouse.set_visible(False)
    fonte_ui = pygame.font.SysFont('Consolas', 28, bold=True)
    
    dialogo_blip_inicio = dialogo_blip_final = som_tiro = som_acerto = som_morte = None

    try:
        dialogo_blip_inicio = pygame.mixer.Sound("blip3.wav")
        dialogo_blip_inicio.set_volume(0.8) 
    except Exception as e: pass

    try:
        dialogo_blip_final = pygame.mixer.Sound("blip.wav")
        dialogo_blip_final.set_volume(0.8) 
    except Exception as e: pass

    try: som_tiro = pygame.mixer.Sound("shoot.wav")
    except: pass
    try: som_acerto = pygame.mixer.Sound("hit.wav")
    except: pass
    try: som_morte = pygame.mixer.Sound("death.wav")
    except: pass

    shader_phong = compileProgram(compileShader(PHONG_VERT, GL_VERTEX_SHADER), compileShader(PHONG_FRAG, GL_FRAGMENT_SHADER))
    shader_crosshair = compileProgram(compileShader(CROSSHAIR_VERT, GL_VERTEX_SHADER), compileShader(CROSSHAIR_FRAG, GL_FRAGMENT_SHADER))
    shader_ui = compileProgram(compileShader(UI_VERT, GL_VERTEX_SHADER), compileShader(UI_FRAG, GL_FRAGMENT_SHADER))
    
    vao_hornet = carregar_modelo_completo("hornet.glb")
    vao_porta = carregar_modelo_completo("door.glb")
    vao_esfera, num_vert_esfera = criar_esfera_vao()
    vao_crosshair, num_vert_crosshair = criar_crosshair_vao()
    vao_ui = criar_ui_quad_vao()
    
    glEnable(GL_DEPTH_TEST); glEnable(GL_CULL_FACE); glLineWidth(2.0)
    projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, largura/altura, 0.1, 100.0)
    
    # --- MATRIZES ---
    escala_h = pyrr.matrix44.create_from_scale([2.0, 2.0, 2.0])
    rot_h = pyrr.matrix44.create_identity() 
    trans_h = pyrr.matrix44.create_from_translation([0.0, -1.0, -5.0])
    model_hornet = pyrr.matrix44.multiply(escala_h, trans_h)
    
    escala_p = pyrr.matrix44.create_from_scale([0.5, 0.5, 0.5])
    rot_p = pyrr.matrix44.create_identity() 
    trans_p = pyrr.matrix44.create_from_translation([0.0, -1.0, -25.0])
    model_porta = pyrr.matrix44.multiply(escala_p, trans_p)
    
    cam_pos, cam_front, cam_up = np.array([0.0, 1.5, 2.0]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])
    yaw, pitch, sensibilidade, speed = -90.0, 0.0, 0.1, 4.0
    pygame.mouse.get_rel() 
    
    current_room = 0 # 0: Dream, 1: Battle, 2: Final
    dialogo_ativo, num_inimigos_vivos = False, 0
    clock = pygame.time.Clock(); running = True

    # --- LISTAS DE DIÁLOGO ---
    dialogos_inicio = [
        "Pequeno inseto, a fundação está corrompida. Para consertar este mundo, você precisa me ajudar.",
        "Atravesse aquela porta. O vazio está cheio de anomalias, bugs e falhas do passado.",
        "Destrua todos os 40 alvos para provar que você é o verdadeiro arquiteto deste lugar!"
    ]
    
    dialogos_final = [
        "Inacreditável... Você conseguiu derrotar todos os 40 ecos do passado.",
        "O caminho até aqui não foi fácil. Foi necessário muita luta e foco inabalável.",
        "Este pequeno jogo é a nossa forma de celebrar os seus 40 anos de vida e história.",
        "É uma alegria gigantesca ter você conosco no VORTEX como nosso coordenador. Feliz Aniversário, Ivo!"
    ]
    
    dialogos_atuais = dialogos_inicio
    char_index_di, time_di, current_di = 0, 0.0, 0
    texto_atual_exibido = ""
    textura_ui = None
    
    inimigos = []
    def gerar_inimigos():
        nonlocal inimigos, num_inimigos_vivos
        inimigos = []
        for _ in range(40):
            r = random.uniform(8, 20)
            theta = random.uniform(0, 2*np.pi)
            phi = random.uniform(0.3, np.pi/2)
            inimigos.append({'position': np.array([r * np.sin(phi) * np.cos(theta), r * np.cos(phi) + 1.0, r * np.sin(phi) * np.sin(theta)]), 'alive': True})
        num_inimigos_vivos = 40

    last_shot_time = 0.0; cooldown_arma = 0.4

    while running:
        dt = clock.tick(60) / 1000.0 
        tempo_atual = pygame.time.get_ticks() / 1000.0 

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
            
            # --- INTERAÇÕES UNIVERSAIS NA TECLA E ---
            elif event.type == KEYDOWN and event.key == K_e:
                dist_h = np.linalg.norm(cam_pos[[0,2]] - np.array([0.0, -5.0]))
                dist_p = np.linalg.norm(cam_pos[[0,2]] - np.array([0.0, -25.0]))
                
                # Sala Inicial
                if current_room == 0:
                    if dist_h < 8.0: 
                        if dialogo_ativo:
                            dialogo_ativo, char_index_di, current_di = False, 0, 0
                        else:
                            dialogos_atuais = dialogos_inicio
                            dialogo_ativo, char_index_di, current_di, texto_atual_exibido = True, 0, 0, ""
                    elif dist_p < 8.0:
                        current_room, cam_pos = 1, np.array([0.0, 1.5, 2.0])
                        gerar_inimigos()
                        
                # Sala Final
                elif current_room == 2:
                    if dist_h < 8.0:
                        if dialogo_ativo:
                            dialogo_ativo, char_index_di, current_di = False, 0, 0
                        else:
                            dialogos_atuais = dialogos_final
                            dialogo_ativo, char_index_di, current_di, texto_atual_exibido = True, 0, 0, ""
            
            # --- TIRO (EXCLUSIVO NO MOUSE - BOTÃO ESQUERDO) ---
            elif event.type == MOUSEBUTTONDOWN and event.button == 1 and current_room == 1:
                if tempo_atual - last_shot_time >= cooldown_arma:
                    last_shot_time = tempo_atual 
                    if som_tiro: som_tiro.play()
                    for ini in inimigos:
                        if ini['alive']:
                            to_enemy = (ini['position'] - cam_pos) / np.linalg.norm(ini['position'] - cam_pos)
                            if np.dot(cam_front, to_enemy) > 0.985 and np.linalg.norm(ini['position'] - cam_pos) < 25.0: 
                                ini['alive'], num_inimigos_vivos = False, num_inimigos_vivos - 1
                                if som_acerto: som_acerto.play()
                                
                                # Terminou de matar os 40 inimigos
                                if num_inimigos_vivos == 0:
                                    if som_morte: som_morte.play()
                                    current_room = 2 
                                    # Teleporta o jogador BEM DE FRENTE para a Hornet
                                    cam_pos = np.array([0.0, 1.5, -2.0]) 
                                    yaw, pitch = -90.0, 0.0
                                break 
        
        if not dialogo_ativo:
            dx, dy = pygame.mouse.get_rel()
            pygame.mouse.set_pos((largura//2, altura//2)) 
            pygame.mouse.get_rel() 
            
            yaw += dx * sensibilidade; pitch -= dy * sensibilidade
            if pitch > 89.0: pitch = 89.0
            if pitch < -89.0: pitch = -89.0
            front = np.array([np.cos(np.radians(yaw)) * np.cos(np.radians(pitch)), np.sin(np.radians(pitch)), np.sin(np.radians(yaw)) * np.cos(np.radians(pitch))])
            cam_front = front / np.linalg.norm(front)
            keys = pygame.key.get_pressed()
            cam_right = np.cross(cam_front, cam_up); cam_right = cam_right / np.linalg.norm(cam_right)
            move_front = np.array([cam_front[0], 0.0, cam_front[2]])
            if np.linalg.norm(move_front) > 0: move_front = move_front / np.linalg.norm(move_front)
            if keys[K_w]: cam_pos += move_front * speed * dt
            if keys[K_s]: cam_pos -= move_front * speed * dt
            if keys[K_a]: cam_pos -= cam_right * speed * dt
            if keys[K_d]: cam_pos += cam_right * speed * dt
            
        glClearColor(1.0, 1.0, 1.0, 1.0); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        view = pyrr.matrix44.create_look_at(cam_pos, cam_pos + cam_front, cam_up)
        
        glUseProgram(shader_phong)
        glUniformMatrix4fv(glGetUniformLocation(shader_phong, "projection"), 1, GL_FALSE, projection)
        glUniformMatrix4fv(glGetUniformLocation(shader_phong, "view"), 1, GL_FALSE, view)
        glUniform3f(glGetUniformLocation(shader_phong, "viewPos"), cam_pos[0], cam_pos[1], cam_pos[2])
        
        # SALA INICIAL
        if current_room == 0:
            glUniformMatrix4fv(glGetUniformLocation(shader_phong, "model"), 1, GL_FALSE, model_hornet)
            for parte in vao_hornet:
                glUniform3f(glGetUniformLocation(shader_phong, "objectColor"), 0.8, 0.1, 0.1) 
                glBindVertexArray(parte['vao']); glDrawArrays(GL_TRIANGLES, 0, parte['count'])
            
            glUniformMatrix4fv(glGetUniformLocation(shader_phong, "model"), 1, GL_FALSE, model_porta)
            for parte in vao_porta:
                glUniform3f(glGetUniformLocation(shader_phong, "objectColor"), 0.5, 0.5, 0.6) 
                glBindVertexArray(parte['vao']); glDrawArrays(GL_TRIANGLES, 0, parte['count'])
            
        # SALA DE BATALHA
        elif current_room == 1:
            for ini in inimigos:
                if ini['alive']:
                    glUniformMatrix4fv(glGetUniformLocation(shader_phong, "model"), 1, GL_FALSE, pyrr.matrix44.create_from_translation(ini['position']))
                    glUniform3f(glGetUniformLocation(shader_phong, "objectColor"), 0.8, 0.1, 0.1)
                    glBindVertexArray(vao_esfera); glDrawArrays(GL_TRIANGLES, 0, num_vert_esfera)

            glUseProgram(shader_crosshair)
            glBindVertexArray(vao_crosshair); glDrawArrays(GL_LINES, 0, num_vert_crosshair)
            
        # SALA FINAL DA HOMENAGEM
        elif current_room == 2:
            glUseProgram(shader_phong)
            glUniformMatrix4fv(glGetUniformLocation(shader_phong, "model"), 1, GL_FALSE, model_hornet)
            for parte in vao_hornet:
                glUniform3f(glGetUniformLocation(shader_phong, "objectColor"), 0.8, 0.1, 0.1) 
                glBindVertexArray(parte['vao']); glDrawArrays(GL_TRIANGLES, 0, parte['count'])
                
            glUseProgram(shader_crosshair)
            glBindVertexArray(vao_crosshair); glDrawArrays(GL_LINES, 0, num_vert_crosshair)

        # LÓGICA DE UI APLICADA ÀS SALAS 0 E 2
        if dialogo_ativo and (current_room == 0 or current_room == 2):
            time_di += dt
            texto_real = dialogos_atuais[current_di]
            
            if time_di > 0.04 and char_index_di < len(texto_real): 
                texto_atual_exibido += texto_real[char_index_di]
                if textura_ui is not None: glDeleteTextures(1, [textura_ui]) 
                textura_ui = gerar_textura_texto(texto_atual_exibido, fonte_ui)
                
                # Escolhe o som correto dependendo da sala
                som_atual = dialogo_blip_inicio if current_room == 0 else dialogo_blip_final
                
                if texto_real[char_index_di] != " " and som_atual is not None:
                    if not pygame.mixer.Channel(0).get_busy():
                        pygame.mixer.Channel(0).play(som_atual)
                
                char_index_di += 1; time_di = 0.0
                
            elif char_index_di == len(texto_real) and time_di > 1.0: 
                if current_di + 1 < len(dialogos_atuais):
                    current_di += 1; char_index_di, time_di, texto_atual_exibido = 0, 0.0, ""
                else: 
                    dialogo_ativo = False 
            
            if textura_ui is not None:
                glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDisable(GL_DEPTH_TEST) 
                glUseProgram(shader_ui); glBindTexture(GL_TEXTURE_2D, textura_ui); glBindVertexArray(vao_ui)
                glDrawArrays(GL_QUADS, 0, 4)
                glEnable(GL_DEPTH_TEST); glDisable(GL_BLEND)

        pygame.display.flip()

    pygame.quit(); sys.exit()

if __name__ == "__main__":
    main()