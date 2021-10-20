clc, clear all, close all, format compact
addpath('/Users/ryanhass/Documents/MATLAB/Lele_Research/matlabFunctions')

% Spatial domain
nx = 64;
ny = 32;
nz = 32;

Lx = 2*pi;
Ly = pi;
Lz = 1.0;

dz = Lz/nz;
z = 0:dz:Lz-dz;

% Time info
tid = 35760;

% Directory paths
homedir = '/Users/ryanhass/Documents/MATLAB/CS_230/Final_project';
datadir = [homedir,'/data/'];
outdir  = [homedir,'/post_processing/'];

% File name pre- and post-fix
tidstr = sprintf('%06d',tid);
fpre = 'Run11_';
fpost = ['_t',tidstr,'.out'];

% Import velocity data
fname = [fpre,'uVel',fpost];
u = read_fortran_box([datadir,fname],nx,ny,nz,'double');
fname = [fpre,'vVel',fpost];
v = read_fortran_box([datadir,fname],nx,ny,nz,'double');
fname = [fpre,'wVel',fpost];
w = read_fortran_box([datadir,fname],nx,ny,nz,'double');

% Import time data
fname = [fpre,'info',fpost];
fid = fopen([datadir,fname]);
t = fscanf(fid,'%f');
time = t(1);

U = squeeze(mean(mean(u,2),1));
up = zeros(nx,ny,nz);
for j = 1:ny
    for i = 1:nx
        up(i,j,:) = squeeze(u(i,j,:)) - U;
    end
end
V = squeeze(mean(mean(v,2),1));
vp = zeros(nx,ny,nz);
for j = 1:ny
    for i = 1:nx
        vp(i,j,:) = squeeze(v(i,j,:)) - V;
    end
end
W = squeeze(mean(mean(w,2),1));
wp = zeros(nx,ny,nz);
for j = 1:ny
    for i = 1:nx
        wp(i,j,:) = squeeze(w(i,j,:)) - W;
    end
end

tke = squeeze(mean(mean(0.5*(up.^2 + vp.^2 + wp.^2),2),1));

uslice_1 = squeeze(u(:,:,2));
uslice_10 = squeeze(u(:,:,11));
uslice_20 = squeeze(u(:,:,21));
uslice_31 = squeeze(u(:,:,end));

figure
subplot(3,1,1)
surface(uslice_1','edgeColor','none'), daspect([1 1 1])
xlim([1 64])
ylim([1 32])
colorbar
subplot(3,1,2)
surface(uslice_10','edgeColor','none'), daspect([1 1 1])
xlim([1 64])
ylim([1 32])
colorbar
subplot(3,1,3)
surface(uslice_20','edgeColor','none'), daspect([1 1 1])
xlim([1 64])
ylim([1 32])
colorbar
% matfile = [outdir,'tke_and_uslices_t',tidstr,'.mat'];
% save(matfile,'tke','time','tid','uslice_1','uslice_10','uslice_20','uslice_31')

figure
subplot(1,2,1)
plot(tke,z)
subplot(1,2,2)
plot(U,z)