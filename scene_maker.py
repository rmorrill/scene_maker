# make a scene with object and image classes 


import pyrender 
import trimesh 
import regex as re 
import ipdb
import os 
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from matplotlib.patches import Rectangle
import json


def ret_filename_no_ext(filename): 
    filesep = os.path.sep
    fparts = re.split(filesep, filename)
    fname_tmp = fparts[-1]
    name = re.split(r'\.', fname_tmp)
    return name[0]

def apply_BW_filter(img): 
    img = img.convert('LA')
    return img
    
def apply_blur_filter(img, blursz): 
    img = img.filter(ImageFilter.GaussianBlur(blursz))
    return img

def apply_hue_rotate(img, huerotate): 
    np_img = np.array(img)
    hsv = cv.cvtColor(np_img, cv.COLOR_RGB2HSV)
    h, s, v = cv.split(hsv)
    rev_h = huerotate + h.astype(int)
    rev_h = np.where(rev_h>179, rev_h-179, rev_h).astype(np.uint8)
    rev_hsv = cv.merge([rev_h, s, v])
    img_huerotate = cv.cvtColor(rev_hsv, cv.COLOR_HSV2RGB)
    new_mat = np.concatenate((img_huerotate, np_img[:,:,[3]]), axis=2)
    img = Image.fromarray(new_mat, 'RGBA')
    return img

def adjust_alpha(img, new_alpha): 
    #img.putalpha(new_alpha)
    r, g, b, alpha = img.split()
    alpha_new = np.where(np.array(alpha)>0, new_alpha, np.array(alpha))
    img = Image.merge('RGBA', (r,g,b,Image.fromarray(alpha_new)))
    return img

def adjust_contrast(img, contrast_adjust): 
    contrast = ImageEnhance.Contrast(img)
    #new_image = img.filter(contrast_adjust)
    img = contrast.enhance(contrast_adjust)
    return img

def get_obj_bounding_box(rgba_img): 
    contours = cv.findContours(rgba_img[:,:,3],  cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    conts_xy = np.array(contours[0], dtype = 'object')
    if len(conts_xy)>1: # multiple BBs
        x, y, w, h = np.zeros((4, len(conts_xy)))
        for ii, cont in enumerate(conts_xy): 
            cont = cont[:].squeeze()
            x[ii], y[ii], w[ii], h[ii] = cv.boundingRect(cont)
            #print(f'x is {x}')
    else: 
        conts_xy = np.array(contours[0])
        cont_xy = conts_xy[:].squeeze()
        x, y, w, h = cv.boundingRect(cont_xy)
    
    #cont_xy = np.array(contours[0])
    #cont_xy = cont_xy[:].squeeze()
    #x, y, w, h = cv.boundingRect(cont_xy)
    return x, y, w, h

def rect_for_crop(rect_sz, crop_sz):

    x_cent = int(rect_sz[0]/2)
    y_cent = int(rect_sz[1]/2)
    x_crop = (x_cent - int(crop_sz[0]/2), x_cent + int(crop_sz[0]/2))
    y_crop = (y_cent - int(crop_sz[1]/2), y_cent + int(crop_sz[1]/2))
    crop_rect = (x_crop[0], y_crop[0], x_crop[1], y_crop[1])
    #breakpoint()
    return crop_rect
    

class StimScene: 
    
    def __init__(self, rendersz = (8e2,8e2), ambient_light_col = [255, 255, 255],\
                bg_col = [128,128,128,0], directional_light_intens = 1e3, maintain_aspect = True): 
        self.maintain_aspect = maintain_aspect # maintain the aspect ratio 
        self.rendersz = rendersz
        self.ambient_light_col = ambient_light_col
        self.bg_col = bg_col
        self.directional_light_intens = directional_light_intens
        self.prepare_scene_lights_camera()
        
    def add_object(self, stim_obj): 
        # calculate pose matrix
        #combined_pose = stim_obj.scale_pose_mat@stim_obj.trans_pose_mat@stim_obj.rotate_pose_mat
        combined_pose = stim_obj.trans_pose_mat@stim_obj.rotate_pose_mat@stim_obj.scale_pose_mat
        self.scene.add(stim_obj.mesh, pose=combined_pose)
        self.stim_obj = stim_obj
        
    def add_background(self, background_obj): 
        #self.background_fname = background_obj.background_fname
        self.background = background_obj
        
    def load_bg_image(self):
        self.bg_img = Image.open(self.background.filename)
        self.bg_img = self.apply_filters_to_img(self.background, self.bg_img)
        return self.bg_img


    def prepare_scene_lights_camera(self):
        # set defaults for camera and light in scene 
        scene = pyrender.Scene(ambient_light=self.ambient_light_col, bg_color=self.bg_col)
        camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0, aspectRatio=1)
        light = pyrender.DirectionalLight(color=[255,255,255], intensity=self.directional_light_intens)
        scene.add(light, pose = np.eye(4))
        scene.add(camera, pose=[[1,  0, 0,  0],
                               [ 0,  1, 0, 0],
                               [ 0,  0, 1,  1000],
                               [ 0,  0,  0,  1]])
        #print('removing light node')
        #scene.remove_node(light)
        self.scene = scene
        return

    def render_fg_img(self): 
        #ipdb.set_trace()
        if not(hasattr(self, 'stim_obj')): 
            self.stim_obj = StimObject()
        rendersz_wd, rendersz_ht = self.rendersz  
        flags = pyrender.RenderFlags.RGBA
        render_max = max(self.rendersz)
        r = pyrender.OffscreenRenderer(render_max, render_max)
        fg_img, _ = r.render(self.scene, flags)
        fg_img = Image.fromarray(fg_img, 'RGBA')
        fg_img_sz = fg_img.size
        
       # breakpoint()
        # apply crop to make it appropriate size 
        
        
        fg_img = self.apply_filters_to_img(self.stim_obj, fg_img)
        return fg_img
        
    
    def compose_scene(self): 
        # get background
        bg_img = self.load_bg_image()
        #breakpoint()
        #
        #ipdb.set_trace()
        #bg_img_AR  =bg_img.size(1)
        bg_img_AR = bg_img.size[0]/bg_img.size[1]
        max_dim = max(bg_img.size)
        max_rsz = max(self.rendersz)
        
        
        if bg_img.size != self.rendersz: 
            #ipdb.set_trace()
            if isinstance(self.rendersz, tuple): 
                rendersz_wd, rendersz_ht = self.rendersz  
            else: 
                rendersz_wd = rendersz_ht = self.rendersz
            
            if self.maintain_aspect:
                scalefact = max_rsz/max_dim
                rsz_wd = bg_img.size[0] * scalefact
                rsz_ht = bg_img.size[1] * scalefact
            else:
                rsz_wd = rendersz_wd
                rsz_ht = rendersz_ht
                
            bg_img = bg_img.resize((int(rsz_wd), int(rsz_ht)))
            
            if self.maintain_aspect: 
                bg_crop_rect = rect_for_crop(bg_img.size, self.rendersz)
                bg_img = bg_img.crop(bg_crop_rect)
    
        fg_img = self.render_fg_img()
        fg_crop_rect = rect_for_crop(fg_img.size, self.rendersz)
        fg_img = fg_img.crop(fg_crop_rect)

        #breakpoint()
        img_combined = Image.alpha_composite(bg_img, fg_img)
        self.fg_img = fg_img
        self.output_image_PIL = img_combined
        self.output_image_array = np.array(img_combined)
        return img_combined
        
    def show_scene(self, figsz = (10,10)): 
        self.compose_scene()
        plt.figure(figsize=figsz), plt.imshow(self.output_image_array)
        self.im_hand = plt.gca()
        #img_combined.show()
        
    def show_object_bounding_box(self, edgecol = (1,0,0), linewidth = 2):
        np_img = np.array(self.fg_img)
        x,y,w,h = get_obj_bounding_box(np_img)
        ax = plt.gca()
        
        if type(x) == int: 
            ax.add_patch(Rectangle((x, y), w, h,
                 edgecolor = edgecol,
                 facecolor = None,
                 fill=False,
                 lw=linewidth))
            plt.title(f'obj w, h: {w}, {h}')
        else: 
            for ii in range(len(x)): 
                 ax.add_patch(Rectangle((x[ii], y[ii]), w[ii], h[ii],
                 edgecolor = edgecol,
                 facecolor = None,
                 fill=False,
                 lw=linewidth))
        
            plt.title(f'obj1 w, h: {w[0]}, {h[0]}')
       
        
    def apply_filters_to_img(self, obj, img): 
        
        # takes an scenelement object with filter properties and a PIL image 
        if obj.BW_filter:
            img = apply_BW_filter(img)
            
        if obj.hue_rotate: 
            img = apply_hue_rotate(img, obj.hue_rotate)    
 
        if obj.gaussian_blur:
            img = apply_blur_filter(img, obj.gaussian_blur) 
            
        if obj.contrast_adj: 
            img = adjust_contrast(img, obj.contrast_adj)
            
        if obj.alpha_adj: 
            img = adjust_alpha(img, obj.alpha_adj)

        if img.mode != 'RGBA': 
            img = img.convert('RGBA')
        return img 
        
        
    def save_scene(self, save_full):
        self.compose_scene()
        self.output_image_PIL.save(save_full)
        print(f'saving image to {save_full}')
        #pass
    
    
class SceneElement(): 
    
    def __init__(self, filename):
        self.filename = filename
        self.gaussian_blur = None
        self.BW_filter = None
        self.hue_rotate = None
        self.contrast_adj = None
        self.alpha_adj = None
        
    def ret_name(self): 
        self.name = ret_filename_no_ext(self.filename)
        return self.name
    
    def set_gaussian_blur(self, gauss_blur_sz): 
        self.gaussian_blur = gauss_blur_sz
        
    def set_BW_filter(self, BW_filter_flag): 
        self.BW_filter = BW_filter_flag
        
    def set_hue_rotate(self, hue_rotation_angle): 
        self.hue_rotate = hue_rotation_angle
        
    def set_contrast(self, contrast_adj):
        self.contrast_adj = contrast_adj
        
    def set_alpha(self, alpha_adj):
        self.alpha_adj = alpha_adj
    
    
class StimObject(SceneElement): 
    
    def __init__(self, filename = '', base_pix_sz = 135):
        super().__init__(filename)
        self.rotate_pose_mat = np.eye(4) # default rotation matrix 
        self.scale_pose_mat = np.eye(4) # default scale matrix 
        self.trans_pose_mat = np.eye(4) # default translation matrix 
        self.x_rot = 0
        self.y_rot = 0
        self.z_rot = 0 
        self.obj_scale = 1
        #self.gaussian_blur = None
        #self.BW_filter = None
        #self.hue_rotate = None
        self.base_pix_sz = base_pix_sz
        self.load_gltf()
        
    def load_gltf(self):
        if self.filename: 
            
            #model = trimesh.load(self.filename)
            #model = model.geometry[list(model.geometry.keys())[1]]
            #self.mesh = pyrender.Mesh.from_trimesh(gltf_tri, smooth=False)
           # breakpoint()
            
            _,fext = os.path.splitext(self.filename)
            if fext == '.glb': 
                #model = trimesh.load(self.filename, file_type = 'glb', process=True)
                model = trimesh.load(self.filename, file_type = 'glb', process=True)
                if len(model.geometry.keys())>1:
                    g = model.geometry
                    g_tmp = ()
                    v = []
                    for k in g: 
                        g_tmp = g_tmp +(g[k])
                        v.append(g[k].visual)
                    
                    vis = trimesh.visual.concatenate(v)
                    model = trimesh.util.concatenate(g_tmp)
                    #model.visual = vis
                else: 
                    g_tmp = model.geometry[list(model.geometry.keys())[0]]
                    vis = g_tmp.visual
                    model = g_tmp
                model.visual = vis
            elif fext == '.obj':  
                model = trimesh.load(self.filename, file_type = 'obj', process=False, force='mesh')
                #breakpoint()
            elif fext == '.mtl':  
                model = trimesh.load(self.filename, file_type = 'obj', process=True, force='mesh')
                
           # breakpoint()
            max_dim = np.max(model.extents)
            print(f'base pix sz is {self.base_pix_sz}')
            #breakpoint()
            # default scaling
            def_scale_fact = (1.4*self.base_pix_sz)/max_dim
            def_scale_matrix = np.eye(4)
            def_scale_matrix[:3, :3] *= def_scale_fact
            model.apply_transform(def_scale_matrix)
            
            # default translation, center of mass to 0,0,0
            centr = model.center_mass 
            trans_mat = np.eye(4)
            trans_mat[:3,3] = centr*-1
            model.apply_transform(trans_mat)
            
            print(model.extents)
            
            self.trimesh = model
            self.mesh = pyrender.Mesh.from_trimesh(model, smooth=False)
        else:
            self.trimesh = None
            self.mesh = None

        
    def rotate_object(self, x_rot, y_rot, z_rot): 
        r = np.deg2rad(np.array([x_rot, y_rot, z_rot]))    
        
        x_pose_mat = np.array([[1, 0., 0., 0.], 
                                [0., np.cos(r[0]), np.sin(r[0]), 0.], 
                                [0., -1*np.sin(r[0]), np.cos(r[0]), 0.], 
                                [0., 0., 0., 1.]])
        
        y_pose_mat = np.array([[np.cos(r[1]), 0., -1*np.sin(r[1]), 0.], 
                                [0., 1., 0., 0.], 
                                [np.sin(r[1]), 0., np.cos(r[1]), 0.], 
                                [0., 0., 0., 1.]])
        
        z_pose_mat = np.array([[np.cos(r[2]),  -1*np.sin(r[2]), 0., 0.], 
                                [np.sin(r[2]), np.cos(r[2]), 0., 0.], 
                                [0., 0., 1., 0.], 
                                [0., 0., 0., 1.]])

        self.rotate_pose_mat = self.rotate_pose_mat@x_pose_mat@y_pose_mat@z_pose_mat
        self.x_rot = self.x_rot + x_rot
        self.y_rot = self.y_rot + y_rot
        self.z_rot = self.z_rot + z_rot 

    def translate_object(self, tx, ty, tz): 
        
        t_mat = np.eye(4)
        t_mat[:3,3] = [tx, ty, tz]
        
        #t_mat = np.array([[1., 0., 0., tx], 
        #                    [0., 1., 0., ty], 
        #                    [0., 0., 1., tz], 
        #                    [0., 0., 0., 1.]])
        
        self.trans_pose_mat = self.trans_pose_mat@t_mat 
        
        
    def scale_object(self, scale_fact): 
        
        #p = np.array([-1.32770572e-03, 2.43623280e+00, 5.40817229e+00])
        #p = np.array([2.00e-04, 3.52e-01, 6.60e+00])
        #base_pix_adj = np.polyval(p, self.base_pix_sz)
        #print(f'bp size: {self.base_pix_sz}, adj: {base_pix_adj}')
        
        #scale_adj = base_pix_adj/self.mesh.scale
        #ipdb.set_trace()
        
        scale_adj = 1
        new_obj_scale = scale_fact * self.obj_scale
        #print(f'new obj scale: {new_obj_scale}')
        s =  scale_adj * new_obj_scale
        
        s_mat = np.eye(4)
        s_mat[:3, :3] *= s
        
        #s_mat = np.array([[s, 0., 0., 0.], 
        #            [0., s, 0., 0.], 
        #            [0., 0., s, 0.], 
        #            [0., 0., 0., 1.]])
        
        self.scale_pose_mat = self.scale_pose_mat@s_mat  
        self.obj_scale = new_obj_scale
    
class BackgroundImg(SceneElement): 
    
    def __init__(self, filename): 
                
        super().__init__(filename)
        
        #self.gaussian_blur = None
        #self.BW_filter = None
        #self.hue_rotate = None
        




        
        

        
        
        
        
        
        
