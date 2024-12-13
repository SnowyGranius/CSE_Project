import gmsh, numpy as np, random

def Model_gmsh(path, name, resolution, shape='ellipse', phi=0.785, hl_ratio=1, size=100, beta=0, random_beta='no'):
    """
    This function meshes a random packing

    path for the right path output
    name of the Closed packing, so it knows where to look
    resolution = mesh size
    shape (='circle') = The shape you want to make the microstructure with, choose between circle, ellipse, triangle and rectangle
    
    phi (=0 radians) = The angle of the point of the triangles, you can describe the sharpness with this
    hl_ratio (=0) = The ratio of the height and length of squares, in which you can go from from perfect squares to any rectangle. 
    size (=100) = percentage which you want to use
    beta (=0 radians) = The angle of the shapes considering the full microstructure
    random_beta (='no') = Random rotation of each shape in the microstructure (set to yes)
    """
    
    gmsh.initialize()
    gmsh.model.add("Random_packing")
    
    #Some variables
    Size = size/100             #Size of the boundary (also works for subsampling)
    Epsilon = 0.000001          #To identify the entities on the boundary
    index = (100-size)/100/2    #this way, we are able to include growing samples from the center of the original random packing.
    size_big = 5                #Ratio of the bigger element size compared to the smaller ones
    distance_max = 3            #Maximum distance of the transition field between large elements and small elements
    distance_min = 2            #Minimum distance of the transition field between large elements and small elements
    
    if shape == 'circle' or shape == 'ellipse':
        shapes = np.genfromtxt('%s/Circle_data/%s_centers.txt' % (path, name))
    else:
        shapes = np.genfromtxt('%s/Polygon_data/%s_centers.txt' % (path, name))
        
    #Inlcude all the shapes, create shape, add loop and surface to make them 2D for cutting them later
    c, w, s = [], [], []
    j=-1
    
    for i in range(len(shapes)):
        if ((shapes[i,3] + shapes[i,0]) > index and (shapes[i,0]-shapes[i,3]) < (Size-index)):
            if ((shapes[i,1] + shapes[i,3]) > index and (shapes[i,1]-shapes[i,3]) < (Size-index)):
                
                x_0 = shapes[i,0]
                y_0 = shapes[i,1]
                r = shapes[i,3]
                r_new = 0.0035 
                
                #For circles:
                if shape == 'circle':
                    j = j+1
                    c.append(gmsh.model.occ.addCircle(shapes[i,0], shapes[i,1], shapes[i,2], shapes[i,3]))
                    w.append(gmsh.model.occ.addCurveLoop([c[j]]))
                    s.append(gmsh.model.occ.add_plane_surface([w[j]]))
                
                #For ellipses:
                if shape == 'ellipse':
                    if random_beta == 'yes':
                            beta = 2*np.pi * random.uniform(0, 1)
                    
                    j = j+1
                    c.append(gmsh.model.occ.addEllipse(shapes[i,0], shapes[i,1], shapes[i,2], shapes[i,3], shapes[i,3]*hl_ratio, zAxis=[0, 0, 1], xAxis=[1, np.tan(beta), 0]))
                    w.append(gmsh.model.occ.addCurveLoop([c[j]]))
                    s.append(gmsh.model.occ.add_plane_surface([w[j]]))
                
                #For triangle:
                elif shape == 'triangle':
                    if r >= r_new:
                        if random_beta == 'yes':
                            beta = 2*np.pi * random.uniform(0, 1)
                        
                        j = j+1
                        x1 = np.cos(beta) * r_new + x_0
                        y1 = np.sin(beta) * r_new + y_0
                        x2 = x1 - np.cos(beta - phi/2) * (2*np.cos(phi/2)*r_new)
                        y2 = y1 + np.sin(phi/2 - beta) * (2*np.cos(phi/2)*r_new)
                        x3 = x1 - np.cos(beta + phi/2) * (2*np.cos(phi/2)*r_new)
                        y3 = y1 - np.sin(beta + phi/2) * (2*np.cos(phi/2)*r_new)
                        
                        point_1 = gmsh.model.occ.addPoint(x1, y1, 0)
                        point_2 = gmsh.model.occ.addPoint(x2, y2, 0)
                        point_3 = gmsh.model.occ.addPoint(x3, y3 ,0)
                        
                        line_1 = gmsh.model.occ.addLine(point_1, point_2)
                        line_2 = gmsh.model.occ.addLine(point_3, point_2)
                        line_3 = gmsh.model.occ.addLine(point_3, point_1)
                        
                        w.append(gmsh.model.occ.addCurveLoop([line_1, line_2, line_3], j))
                        s.append(gmsh.model.occ.add_plane_surface([w[j]]))
                        
                        
                #For square:
                elif shape == 'rectangle':
                    if r >= r_new:
                        if random_beta == 'yes':
                            beta = 2*np.pi * random.uniform(0, 1)
                        
                        j = j+1
                        x1 = x_0 - r_new * np.cos(np.arctan(hl_ratio) - beta)
                        y1 = y_0 + r_new * np.sin(np.arctan(hl_ratio) - beta)
                        x2 = x_0 + r_new * np.cos(np.arctan(hl_ratio) + beta)
                        y2 = y_0 + r_new * np.sin(np.arctan(hl_ratio) + beta)
                        x3 = x_0 + r_new * np.cos(np.arctan(hl_ratio) - beta)
                        y3 = y_0 - r_new * np.sin(np.arctan(hl_ratio) - beta)
                        x4 = x_0 - r_new * np.cos(np.arctan(hl_ratio) + beta)
                        y4 = y_0 - r_new * np.sin(np.arctan(hl_ratio) + beta)
                        
                        point_1 = gmsh.model.occ.addPoint(x1, y1, 0)
                        point_2 = gmsh.model.occ.addPoint(x2, y2, 0)
                        point_3 = gmsh.model.occ.addPoint(x3, y3, 0)
                        point_4 = gmsh.model.occ.addPoint(x4, y4, 0)
                        
                        line_1 = gmsh.model.occ.addLine(point_1, point_2)
                        line_2 = gmsh.model.occ.addLine(point_2, point_3)
                        line_3 = gmsh.model.occ.addLine(point_3, point_4)
                        line_4 = gmsh.model.occ.addLine(point_4, point_1)
                        
                        w.append(gmsh.model.occ.addCurveLoop([line_1, line_2, line_3, line_4], j))
                        s.append(gmsh.model.occ.add_plane_surface([w[j]]))
                        
                        
    gmsh.model.occ.synchronize()
    
    #Create the rectangle,
    r = gmsh.model.occ.addRectangle(index,index,0,(Size-2*index),(Size-2*index))
    b = gmsh.model.occ.cut([(2,r)], [(2,elt) for elt in s])
    gmsh.model.occ.synchronize()

    #Identify all the entitites
    b_top = gmsh.model.getEntitiesInBoundingBox(index-Epsilon, (Size-index-Epsilon), -Epsilon, (Size-index+Epsilon), (Size-index+Epsilon), Epsilon, dim=1)
    b_left = gmsh.model.getEntitiesInBoundingBox((index-Epsilon), index-Epsilon, -Epsilon, (index+Epsilon), (Size-index+Epsilon), Epsilon, dim=1)
    b_right = gmsh.model.getEntitiesInBoundingBox((Size-index-Epsilon), index-Epsilon, -Epsilon, (Size-index+Epsilon), (Size-index+Epsilon), Epsilon, dim=1)
    b_bottom = gmsh.model.getEntitiesInBoundingBox(index-Epsilon, (index-Epsilon), -Epsilon, (Size-index+Epsilon), (index+Epsilon), Epsilon, dim=1) 
    grains_1 = gmsh.model.getEntitiesInBoundingBox((index-Epsilon), (index-Epsilon), -Epsilon, (Size-index+Epsilon), (Size-index+Epsilon), Epsilon, dim=1) 
    box = gmsh.model.getEntitiesInBoundingBox((index-Epsilon), (index-Epsilon), -Epsilon, (Size-index+Epsilon), (Size-index+Epsilon), Epsilon, dim=2)
    
    #Loop to get all the curves within the box, but not the boundaries
    grains = []
    temp = b_top + b_right + b_bottom + b_left
    for k in grains_1:
        if k not in temp:
            grains.append(k) 
    
    #Add the physical groups and tags to the elements
    gmsh.model.addPhysicalGroup(dim=1, tags = ([seq[1] for seq in grains]), tag = 6)
    gmsh.model.setPhysicalName(1, 6, "grains")
    gmsh.model.addPhysicalGroup(dim=1, tags = ([seq[1] for seq in b_top]), tag = 4)
    gmsh.model.setPhysicalName(1, 4, "top")
    gmsh.model.addPhysicalGroup(dim=1, tags = ([seq[1] for seq in b_left]), tag = 5)
    gmsh.model.setPhysicalName(1, 5, "left")
    gmsh.model.addPhysicalGroup(dim=1, tags = ([seq[1] for seq in b_right]), tag = 3)
    gmsh.model.setPhysicalName(1, 3, "right")
    gmsh.model.addPhysicalGroup(dim=1, tags = ([seq[1] for seq in b_bottom]), tag = 2)
    gmsh.model.setPhysicalName(1, 2, "bottom")
    gmsh.model.addPhysicalGroup(dim=2, tags = ([seq[1] for seq in box]), tag = 1)
    gmsh.model.setPhysicalName(2, 1, "block")
    
    #Identify different fields for different mesh size:
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [seq[1] for seq in grains])
    
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", resolution)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", resolution * size_big)
    gmsh.model.mesh.field.setNumber(2, "DistMin", distance_min * resolution)
    gmsh.model.mesh.field.setNumber(2, "DistMax", distance_max * resolution)

    gmsh.model.mesh.field.add("Min", 3)
    gmsh.model.mesh.field.setNumbers(3, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(3)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    
    #Generate and save mesh
    gmsh.model.mesh.generate()
    gmsh.write('%s/Meshes/%s_%s_res_%s_3_mesh_2d.msh' %(path, name, shape, resolution))
    gmsh.finalize()
