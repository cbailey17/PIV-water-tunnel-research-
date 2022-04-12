%% Matlab script For SDSU AE499 Dr. X. Liu
% Author: Alex Bailey
% 1. Generates an icosahedron and performs n sudivisions in order to create 
%    equidistant points on a sphere
% 2. creates parallel rays to a chosen vertex or centroid and determines
%    entrance and exit intersection points with the spheres inscribed cuboid

%% Create an icosahedron with 20 faces and 12 vertices
% constants
% tau: the golden ratio
% n: number of subdivisions
n = 2;
tau = (1+sqrt(5)) / 2;

% create the 12 vertices using known coordinate relationships (x, y, z)
v = [-1, tau, 0; 
      1, tau, 0; 
     -1,-tau, 0; 
      1,-tau, 0; 
      0,-1, tau; 
      0, 1, tau; 
      0,-1,-tau; 
      0, 1,-tau; 
      tau, 0,-1; 
      tau, 0, 1; 
     -tau, 0,-1; 
     -tau, 0, 1]; 
 
% Euclidean normalize vertices to unit size to project the vertices onto a sphere
v = v./norm(v(1,:));

% create 20 faces of the icosahedron
f = [ 1,12, 6; 
      1, 6, 2; 
      1, 2, 8; 
      1, 8,11; 
      1,11,12; 
      2, 6,10; 
      6,12, 5; 
     12,11, 3; 
     11, 8, 7; 
      8, 2, 9; 
      4,10, 5; 
      4, 5, 3; 
      4, 3, 7; 
      4, 7, 9; 
      4, 9,10; 
      5,10, 6; 
      3, 5,12; 
      7, 3,11; 
      9, 7, 8; 
     10, 9, 2]; 
 

%% Subdivide icosahedron in order to uniformly distribute dots on sphere
% tic;
% recursively subdivide equilateral triangle faces
for i = 1:n
    newF = zeros(size(f,1)*4,3);
    for j = 1:size(f,1) % for each triangle
        tri = f(j,:);
        % calculate mid points (add new points to v)
        [a,v] = bisectTriangle(tri(1),tri(2),v);
        [b,v] = bisectTriangle(tri(2),tri(3),v);
        [c,v] = bisectTriangle(tri(3),tri(1),v);
        % generate new subdivision triangles
        newFace = [tri(1),a,c;
                   tri(2),b,a;
                   tri(3),c,b;
                   a,b,c];
        % replace triangle with subdivision
        index = 4*(j-1)+1:4*j;
        newF(index,:) = newFace;
    end
    f = newF;  
end
% toc;

% remove duplicate vertices that occurr when sudiving adjacent triangles
[v,b,ix] = unique(v,'rows'); 

% reassign faces to trimmed vertex list and remove any duplicate faces
f = unique(ix(f),'rows');

% Plot subdived icosahedron
s = patch('Faces',f,'Vertices',v, 'VertexNormals',v,'LineWidth',0.5, ...
    'FaceLighting','phong', 'BackFaceLighting','unlit', ...
    'AmbientStrength', 0.3,'DiffuseStrength',0.6, 'SpecularExponent', ...
    10,'SpecularStrength',0.9, 'FaceColor','flat','CData',v(:,3));
alpha(s, 0.08) 
hold on;
title('Sudivided Icosahedron', 'FontSize', 16);

%% calculations
%Compute radius and length width and height (l)
r = sqrt((v(1,1))^2 + (v(1,2))^2 + (v(1,3)^2));
l = sqrt((r/3));

%calls to functions in order to visualize results
plotCube(l);
%vector2vertex(v(1,1),v(1,2),v(1,3));
%faceCentroid(f, 10, v, 1);
%gridRays(v,1,0.3); hold on;

%% functions written to facilitate project goals
function [dist] = distance(i, j, v)
% Distance formula to find distance between vertices and ensure uniformity 
% input: i = first vertex 
%        j = second vertex
%        v = vertex data
% output: dist = distance between two vertices

dist = sqrt((v(i,1)-v(j,1))^2 + (v(i,2)-v(j,2))^2 + (v(i,3)-v(j,3))^2);
end

function [i,v] = bisectTriangle(t1,t2,v)
% function: bisectTriangle is used in the subdivision of the icosahedron
%           faces. This function bisects the triangles sides in order to
%           develop more equilateral triangles from a single triangle
% input:  t1 = a desired point
%         t2 = a second point that connects to the first
%         v = the current list of vertices
% output: i = the new size of the list of vertices
%         v = the new list of vertices
p1 = v(t1,:); 
p2 = v(t2,:);

% calculate mid point and normalize
pm = (p1 + p2) ./ 2;
pm = pm./norm(pm);

% add to vertices list, return index
i = size(v,1) + 1;
v = [v;pm];
end

function [cords] = faceCentroid(face, cf, v, plot)
% faceCentroid is a function to find the centroid of any given face. This 
% allows for more flexibilty when developing parallel rays
% input:  f = face data
%         v = vertex data 
%         cf = any desired/chosen face number 
%         plot = boolean meant to let user choose to plot a vector to the
%                centroid 1 = plot 0 = dont plot
% output: cords = the cartesian coordinate values of the centroid 

v1 = v(face(cf,1),:);
v2 = v(face(cf,2),:);
v3 = v(face(cf,3),:);

% Calculate mean of given vertices of the face
meanx = mean([v1(1) v2(1) v3(1)]);
meany = mean([v1(2) v2(2) v3(2)]);
meanz = mean([v1(3) v2(3) v3(3)]);

% Normalize the centroid in order to project onto the sphere
centroid = [meanx meany meanz] ./ norm([meanx meany meanz]);

% Plot the vector to the centroid
if plot == 1
vector2vertex(centroid(1), centroid(2), centroid(3)); hold on;
end
cords = centroid;
return 
end

function [] = vector2vertex(x, y, z)
% vector2vertex plots a vector from origin to a given vertex
% input: x y z coordinates of any desired point
origin = [0 0 0];
vector = [x y z];
plot3([origin(1) vector(1)], [origin(2) vector(2)],[origin(3) vector(3)], 'b*-', 'LineWidth', 1);
end
 
function [] = plotCube(l)
% function that plots an inscribed cuboid into the sphere in order to view
% intersection points. This function was written for visualization purposes
% Inputs: l - the length from the origin to the faces of the cube. It is
%         also 1/2 of the width, height, and length
% Outputs: None
c1 = [l, l, l];
c2 = [-l, -l, -l];
c3 = [-l, l, l];
c4 = [l, -l, l];
c5 = [l, l, -l];
c6 = [l, -l, -l];
c7 = [-l, -l, l];
c8 = [-l, l, -l];
plot3([c1(1) c2(1) c3(1) c4(1) c5(1) c6(1) c7(1) c8(1)], ...
    [c1(2) c2(2) c3(2) c4(2) c5(2) c6(2) c7(2) c8(2)], ...
    [c1(3) c2(3) c3(3) c4(3) c5(3) c6(3) c7(3) c8(3)], 'r*'); hold on;
plot3([c1(1) c4(1)], [c1(2) c4(2)], [c1(3) c4(3)], 'r*-', 'LineWidth', 1.2);
plot3([c1(1) c3(1)], [c1(2) c3(2)], [c1(3) c3(3)], 'r*-', 'LineWidth', 1.2);
plot3([c1(1) c5(1)], [c1(2) c5(2)], [c1(3) c5(3)], 'r*-', 'LineWidth', 1.2);
plot3([c5(1) c8(1)], [c5(2) c8(2)], [c5(3) c8(3)], 'r*-', 'LineWidth', 1.2);
plot3([c5(1) c6(1)], [c5(2) c6(2)], [c5(3) c6(3)], 'r*-', 'LineWidth', 1.2);
plot3([c6(1) c2(1)], [c6(2) c2(2)], [c6(3) c2(3)], 'r*-', 'LineWidth', 1.2);
plot3([c2(1) c8(1)], [c2(2) c8(2)], [c2(3) c8(3)], 'r*-', 'LineWidth', 1.2);
plot3([c8(1) c3(1)], [c8(2) c3(2)], [c8(3) c3(3)], 'r*-', 'LineWidth', 1.2);
plot3([c2(1) c7(1)], [c2(2) c7(2)], [c2(3) c7(3)], 'r*-', 'LineWidth', 1.2);
plot3([c7(1) c3(1)], [c7(2) c3(2)], [c7(3) c3(3)], 'r*-', 'LineWidth', 1.2);
plot3([c6(1) c4(1)], [c6(2) c4(2)], [c6(3) c4(3)], 'r*-', 'LineWidth', 1.2); 
plot3([c7(1) c4(1)], [c7(2) c4(2)], [c7(3) c4(3)], 'r*-', 'LineWidth', 1.2); hold on;
end

function [] = plotNormals()
%plotNormals when called will plot normal vectors to each face of the
%spheres inscribed cuboid. It needs no input and does not return anything
% this function was written for visualization and debugging purposes
n1 = [1,0,0];
n2 = [0,1,0];
n3 = [0,0,1];
n4=[-1,0,0];
n5=[0,-1,0];
n6=[0,0,-1];
vector2vertex(n1(1), n1(2), n1(3));
vector2vertex(n2(1), n2(2), n2(3));
vector2vertex(n3(1), n3(2), n3(3));
vector2vertex(n4(1), n4(2), n4(3));
vector2vertex(n5(1), n5(2), n5(3));
vector2vertex(n6(1), n6(2), n6(3));
end

function [] = gridRays(v, pt, density)
% function: gridRays creates a normal grid of hexagonal points of a
%           desired density in which our parallel rays will be created
%           from. The intersection points are then calculated and
%           displayed in the command window

% input: v = vertex data gathered from subdivisions
%        pt = desired point about which to create normal plane
%        density = desired density of the hexagonal grid. A higher value
%                  gives the grid more dense points and thus rays


% The following sectioned code is optional for visualization purposes
% ------------------------------------------------------------------------
% plot vector from origin to chosen vertex to ensure orthonormality.
% uncomment following line for visualization
% vector2vertex(v(pt,1),v(pt,2),v(pt,3));

% Plot surface normal to chosen point for visualization purposes
% uncomment this portion of code to visualize a surface normal to the point
% [x1, y1] = meshgrid(-1:.1:1);
% for i =1:length(x1)
% x2 = v(pt,1) + n(1,1)*y11 + n(1,2)*x11;
% y2 = v(pt,2) + n(2,1)*y11 + n(2,2)*x11;
% z2 = v(pt,3) + n(3,1)*y11 + n(3,2)*x11;
% plot3(v(pt,1),v(pt,2),v(pt,3), 'b*'); hold on;
% plot3(x(i),y(i),z(i), 'b*');
% end
% surf(x2,y2,z2)
% ------------------------------------------------------------------------

%define bounds of the grid
point_X = -5:3:5;
point_Y = -5:sqrt(3):5;
%define variables
d = 1;
hex_X = [];
hex_Y = [];
x_mid_in = [];
y_mid_in = [];
x_mid_out = [];
y_mid_out = [];
k = 0;

%determine the points for the hexagonal grid
for j = 1:1:length(point_X)
    for q = 1:1:length(point_Y)        
        hex_X_temp = [];
        hex_Y_temp = [];        
        kk = 0;
        %iterate through the full hexagon by degrees
        for i = [60,120,180,240,300,360,60]            
            k = k+1;
            kk = kk + 1;            
            hex_01_x = d * cosd(i);
            hex_01_y = d * sind(i);            
            hex_X(k,:) = point_X(j) + hex_01_x;
            hex_Y(k,:) = point_Y(q) + hex_01_y;           
            hex_X_temp(kk,:) = point_X(j) + hex_01_x;
            hex_Y_temp(kk,:) = point_Y(q) + hex_01_y;
        end                 
            x1 = [hex_X_temp(1) hex_X_temp(4)]; 
            y1 = [hex_Y_temp(1) hex_Y_temp(4)];
            
            % Line2 (second and fifth point of hexagon - in counterclockwise)
            x2 = [hex_X_temp(2) hex_X_temp(5)];
            y2 = [hex_Y_temp(2) hex_Y_temp(5)];           
            %fit linear polynomial
            p1 = polyfit(x1,y1,1);
            p2 = polyfit(x2,y2,1);            
            %calculate intersection
            x_intersect = fzero(@(x) polyval(p1-p2,x),3);
            y_intersect = polyval(p1,x_intersect);            
            x_mid_in = [x_mid_in ; x_intersect];
            y_mid_in = [y_mid_in ; y_intersect];
            % Line 3 (first and second point of hexagon - in counterclockwise)
            x3 = [hex_X_temp(1) hex_X_temp(2)]; 
            y3 = [hex_Y_temp(1) hex_Y_temp(2)];          
            % Line 4 (fifth and sixth point of hexagon - in counterclockwise)
            x4 = [hex_X_temp(6) hex_X_temp(5)];
            y4 = [hex_Y_temp(6) hex_Y_temp(5)];          
            %fit linear polynomial
            p3 = polyfit(x3,y3,1);
            p4 = polyfit(x4,y4,1);            
            %calculate intersection
            x_mid = fzero(@(x) polyval(p3-p4,x),3);
            y_mid = polyval(p3,x_mid);
            
            x_mid_out = [x_mid_out ; x_mid];
            y_mid_out = [y_mid_out ; y_mid];
    end
end
% concatenate calculated points for plotting purposes
x1 = [hex_X; x_mid_in; x_mid_out;]; 
y1 = [hex_Y; y_mid_in; y_mid_out;];
% multiply points by scaling factor in order to determine density of grid
x1 = x1*density; y1 = y1*density;

% calculate the gradient and null matrix for defining the normal plane
grad = 2*[v(pt,1) v(pt,2) v(pt,3)];
n = null(grad);

%PLot hexagonal grid arrangement on normal plane.
for i =1:length(x1)
    x(i) = v(pt,1) + n(1,1)*y1(i) + n(1,2)*x1(i);
    y(i) = v(pt,2) + n(2,1)*y1(i) + n(2,2)*x1(i);
    z(i) = v(pt,3) + n(3,1)*y1(i) + n(3,2)*x1(i);
    plot3(v(pt,1),v(pt,2),v(pt,3), 'y*'); hold on;
    plot3(x(i),y(i),z(i), 'y*');
end

%Plot parallel rays
for i = 1:length(x)
    t = -2:0;
    x22 = x(i)+v(pt,1)*t;
    y22 = y(i)+v(pt,2)*t;
    z22 = z(i)+v(pt,3)*t;
    ray = plot3(x22,y22,z22, 'y-', 'LineWidth', 0.7); hold on; ray.Color(4)=0.3;
end

% Intersection points
l=0.5774;
ip = [];
intPoints = [];
plotCube(l);

for i = 1:length(x)
% Determine end point of the rays for finding intersection points
xEnd = x(i)+v(pt,1)*-2;
yEnd = y(i)+v(pt,2)*-2;
zEnd = z(i)+v(pt,3)*-2;

% Determine intersection points of the rays and the cuboid
I1 = intersectionPoints([1 0 0], [l 0 0], [x(i) y(i) z(i)],[xEnd,yEnd,zEnd]);
I2 = intersectionPoints([0 1 0], [0 l 0], [x(i) y(i) z(i)],[xEnd,yEnd,zEnd]);
I3 = intersectionPoints([0 0 1], [0 0 l], [x(i) y(i) z(i)],[xEnd,yEnd,zEnd]);
I4 = intersectionPoints([-1 0 0], [-l 0 0], [x(i) y(i) z(i)],[xEnd,yEnd,zEnd]);
I5 = intersectionPoints([0 -1 0], [0 -l 0], [x(i) y(i) z(i)],[xEnd,yEnd,zEnd]);
I6 = intersectionPoints([0 0 -1], [0 0 -l], [x(i) y(i) z(i)],[xEnd,yEnd,zEnd]);

% ensure the intersection points lie within the bounds of the cuboid faces
if (I1(3)>-l && I1(3)<l && I1(2)<l && I1(2)>-l)  
    ip = [ip;I1];
    intPoints = [intPoints;I1];
end
if (I2(1)>-l && I2(1)<l && I2(3)<l && I2(3)>-l)  
    ip = [ip;I2];
    intPoints = [intPoints;I2];
end
if (I3(1)>-l && I3(1)<l && I3(2)<l && I3(2)>-l)  
    ip = [ip;I3];    
    intPoints = [intPoints;I3];    
end
if (I4(3)>-l && I4(3)<l && I4(2)<l && I4(2)>-l)  
    ip = [ip;I4];
    intPoints = [intPoints;I4];    
end
if (I5(1)>-l && I5(1)<l && I5(3)<l && I5(3)>-l)  
    ip = [ip;I5];
    intPoints = [intPoints;I5];    
end
if (I6(1)>-l && I6(1)<l && I6(2)<l && I6(2)>-l)  
    ip = [ip;I6];
    intPoints = [intPoints;I6];
end
disp("The intersection points of the parallel rays are: ");
disp(intPoints)

%determine the distance from the intersection point to the beginning of the
%ray in order to distinguish between entrance and exit points
if length(ip) >2
    d1 = sqrt((ip(1,1)-x(i))^2+(ip(1,2)-y(i))^2+(ip(1,3)-z(i))^2);
    d2 = sqrt((ip(2,1)-x(i))^2+(ip(2,2)-y(i))^2+(ip(2,3)-z(i))^2);
if d1>d2
    plot3(ip(1,1),ip(1,2),ip(1,3),'g+','MarkerSize', 6); hold on;
    plot3(ip(2,1),ip(2,2),ip(2,3),'r^','MarkerSize', 6); hold on;
    %disp("d2 is entry point");
else 
    plot3(ip(1,1),ip(1,2),ip(1,3),'r^','MarkerSize', 6); hold on;
    plot3(ip(2,1),ip(2,2),ip(2,3),'g+','MarkerSize', 6); hold on;
    %disp("d1 is entry point");
end   
end
ip = [];
end
end

function [I] = intersectionPoints(n, V0, P0, P1)
% intersectionPoints computes the intersection of a plane and the parallel rays
% Inputs: 
%       n: normal vector of the Plane 
%       V0: any point on the Plane 
%       P0: end point 1 of the line P0P1
%       P1: end point 2 of the line P0P1
% Outputs:
%      I: is the point of interection 
u = P1-P0;
w = P0-V0;
D = dot(n,u);
N = -dot(n,w);
sI = N / D;
I = P0+ sI.*u;
end

