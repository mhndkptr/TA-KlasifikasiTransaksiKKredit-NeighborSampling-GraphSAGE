classdef FIGRL
    properties
        intermediate_dim
        final_dim
    end
    methods
        function this = FIGRL(intermediate_dim, final_dim)
            this.intermediate_dim = intermediate_dim;
            this.final_dim = final_dim;
        end
        function [U, Sigma, V] = train_step_figrl(obj, edges)
            Edgelist = unique(sort(edges,2),'rows');
            G = graph(Edgelist(:,1),Edgelist(:,2),'OmitSelfLoops');
            N = numnodes(G);
            A = adjacency(G);
            D_inv = spdiags(1./degree(G),0,N,N);
            Normalized_random_walk = sqrt(D_inv)*A*sqrt(D_inv);
            S = randn(N, obj.intermediate_dim) / sqrt(obj.intermediate_dim);
            C = Normalized_random_walk * S;
            [U,Sigma,V] = svds(C,obj.final_dim,'largest');
        end
        function U = inductive_step_figrl(obj, edges, Sigma, V)
            Edgelist = unique(sort(edges,2),'rows');
            G = graph(Edgelist(:,1),Edgelist(:,2),'OmitSelfLoops');
            N = numnodes(G);
            A = adjacency(G);
            D_inv = spdiags(1./degree(G),0,N,N);
            Normalized_random_walk = sqrt(D_inv)*A*sqrt(D_inv);
            S = randn(N, obj.intermediate_dim) / sqrt(obj.intermediate_dim);
            C = Normalized_random_walk*S;
            U = C*V*inv(Sigma);
            U = sqrt(D_inv)*U;
        end
    end
end

