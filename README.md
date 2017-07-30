# p-Spectral Clustering

This archive contains a Matlab implementation of *p-Laplacian based 
spectral clustering*. Given a graph with weight matrix W, a bipartition 
is computed using the second eigenvector of the unnormalized or 
normalized graph p-Laplacian. A multipartitioning is then obtained using 
a recursive splitting scheme.



## Installation

To install p-Spectral Clustering, compile the mexfiles 
by starting the make.m script from within Matlab. The clustering can 
then be computed using the function 'pSpectralClustering'. 



### Usage

    [clusters,cuts,cheegers] = pSpectralClustering(W,p,normalized,k)

### Input variables

    W             Sparse weight matrix. Has to be symmetric.
    p             Has to be in the interval ]1,2]. Controls the trade-off 
                  between a relaxation of Rcut/Ncut (p=2) and RCC/NCC (p->1)
    normalized    true for Ncut/NCC, false for Rcut/RCC
    k             number of clusters

### Output variables

    clusters      mx(k-1) matrix containing in each column the computed 
                  clustering for each partitioning step.
    cuts          (k-1)x1 vector containing the Ratio/Normalized Cut values 
                  after each partitioning step.
    cheegers      (k-1)x1 vector containing the Ratio/Normalized Cheeger 
                  Cut values after each partitioning step.

For more information type 'help functionname' on the Matlab prompt.



## References

    @inproceedings{BueHei2009,
      author ={B\"{u}hler, Thomas and Hein, Matthias},
      title = {Spectral {C}lustering based on the graph $p$-{L}aplacian},
      booktitle = {Proceedings of the 26th International Conference on Machine Learning},
      pages={81-88},
      publisher={Omnipress},
      year={2009}
    }



## License

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

If you use this code for your publication, please include a reference 
to the paper "Spectral Clustering based on the graph p-Laplacian".



## Contact

Thomas Bühler and Matthias Hein (tb/hein@cs.uni-saarland.de). 
Machine Learning Group, Saarland University, Germany (http://www.ml.uni-saarland.de).
