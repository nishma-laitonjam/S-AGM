% Taken from the paper "Visualization of communities in networks" 
% (http://netwiki.amath.unc.edu/VisComms/VisComms)

function graphplot2d(xy,W,weight,varargin)
% This program was written to take in xy-coordinates of a network as well as
% the adjacency matrix W, to graph the network using these coordinates.  We
% have many programs that can calculate these coordinates either respecting
% community structure (KamadaC.m, fruc_reinC.m, KKFR.m, FRKK.m) or ignoring 
% it (Kamada.m, fruc_rein.m).  This program can also take in a factor for edge length, 
% colors for each node and shapes for each node.  If colors and alpha are given only,
% this program sets up the skeleton for a very nice legend.
%
%Inputs:
%
% xy= matrix of xy-coordinates for each node in the network
%
% W=Adjacency Matrix for the network
%
% Optional Variables:
% 
% alpha = factor for edge length
%
% colors = a vector of numbers defining colors, if zero is one of the
% colors, that is given as purple
%
% shapes = a vector of strings defining the shapes for each node, i. e. 'd'
% for diamond, '.' for dot, etc.
%
% Function Calls:
%
% graphplod2d(xy,W)
% graphplot2d(xy,W,alpha)
% graphplot2d(xy,W,alpha,colors)
% graphplot2d(xy,W,alpha,colors,shapes)
%
% Last Modified by ALT May 19,2009, Created by PJM


if (length(varargin))
    alpha=varargin{1};
else
    alpha=2;
end

%
%Set colormap by scores, if a vector (assumed to be of correct length)
% map = colormap;
map = rand(20,3);
% set a new map of size 20
map(1, :) = [1 1 0]; % yellow
map(2, :) = [1 0 1]; % magenta (pink)
map(3, :) = [0 1 1]; % cyan (greenish-blue)
map(4, :) = [1 0 0]; % red
map(5, :) = [0 1 0]; % green
map(6, :) = [0 0 1]; % Blue
map(7, :) = [0.5020 0.5020 0]; % Olive
map(8, :) = [0.5020 0 0]; % Maroon
map(9, :) = [0.61 0.51 0.74]; % Purple
map(10, :) = [0.9292 0.3585 0.4058]; % lightRed
map(11, :) = [0.8955 0.7398 0.2344]; % DarkYellow
map(12, :) = [0 0 0]; % Black
% map(20, :) = [0 0 0]; % Black

% disp(map);

if (length(varargin)>1)
    scores=varargin{2};
    if weight == 0
        Rcolor = scores-min(scores)+1;
    else
        Ucolor = map(ceil(scores-min(scores)+1),:);
    end
end

if (length(varargin)>2)
    shapes=varargin{3};
else
    shape='.';
    shapes=repmat(shape,length(W),1);
    
end
%
x=xy(:,1); y=xy(:,2);
edges=find(W);

We=[W(edges),edges];
sortWe=sortrows(We);


% This is for a Weighted Network or an unweighted network
% str=(W/max(max(W))).^alpha;

% This is for Making the edges random strengths
 str=rand(size(W));


hold on
N=length(W);
for ie=sortWe(:,2)'
    i=mod(ie-1,N)+1;
    j=floor((ie-1)/N)+1;
    
    if (j>i)
        h=plot(x([i,j]),y([i,j])); hold on;
%         set(h,'color',zeros(1,3));
        set(h,'color',str(i,j)*ones(1,3))
        set(h,'LineWidth',.6);
        %set(h,'color',[.2 .2 .2]);
        
    end
end
if (length(varargin)>1)
    for i=1:N
        if shapes(i,:)=='.'
            if weight == 0
                h=plot(x(i),y(i),shapes(i,:),'markersize',40);
                set(h,'MarkerFaceColor',map(ceil(Rcolor(i)),:));
                set(h,'Color',map(ceil(Rcolor(i)),:));
            else
                percents = weight(i,:);
                if ((i == 59)||(i == 15)||(i == 35))
                    radius=0.06;
                else
                    radius=0.03;
                end
                drawpie(percents,xy(i,:),radius,Ucolor);
            end
        end
    end
    for i=1:N
        if shapes(i,:)~='.'
        h=plot(x(i),y(i),shapes(i,:),'markersize',8);
        set(h,'MarkerFaceColor',[.7 .7 .7]);
        set(h,'Color',map(ceil(Rcolor(i)),:));
        end
    end
else
    plot(x,y,'b.','markersize',20)
end

%

axis equal
hold off
end
function drawpie(percents,pos,radius,colors)



points = 40;
x = pos(1);
y = pos(2);
last_t = 0;
if (length(find(percents))>1)
    for i = 1:length(percents)
        end_t = last_t + percents(i)*points;
        tlist = [last_t ceil(last_t):floor(end_t) end_t];
        xlist = [0 (radius*cos(tlist*2*pi/points)) 0] + x;
        ylist = [0 (radius*sin(tlist*2*pi/points)) 0] + y;
        patch(xlist,ylist,colors(i,:))
        last_t = end_t;
    end
else
    i=find(percents);
    tlist = [0:points];
    xlist = x+radius*cos(tlist*2*pi/points);
    ylist = y+radius*sin(tlist*2*pi/points);
    patch(xlist,ylist,colors(i,:))
end
end
