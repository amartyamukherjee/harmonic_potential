from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import (fem, mesh, plot)
from dolfinx.fem import (Constant, Function, FunctionSpace, 
                         assemble_scalar, dirichletbc, form, locate_dofs_geometrical, locate_dofs_topological)
from petsc4py.PETSc import ScalarType
from dolfinx import geometry
import pyvista


class Quadrotor2DHarmonicPotential:
    def __init__(
            self,
            domain_corners,
            unsafe_boxes,
            target_box,
            c_goal=0.0,
            c_unsafe=1.0,
            f_val=0,
            num_triangles_x=50,
            num_triangles_y=50
        ):
        # Create the triangle mesh
        self.domain = mesh.create_rectangle(MPI.COMM_WORLD,
                                    domain_corners,
                                    n=[num_triangles_x, num_triangles_y])

        V = FunctionSpace(self.domain, ("CG", 1))
        self.V = V

        print("Solving the variational problem...")
        bcs = []

        tdim = self.domain.topology.dim
        fdim = tdim - 1
        self.domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(self.domain.topology)

        self.bb_tree = geometry.BoundingBoxTree(self.domain, self.domain.topology.dim)

        boundary_dofs = locate_dofs_topological(V, fdim, boundary_facets)

        u_L = Function(V)
        u_L.interpolate(lambda x: c_unsafe + 0*x[1])

        bc_L = dirichletbc(u_L, boundary_dofs)
        bcs.append(bc_L)

        for box in unsafe_boxes:
            dofs_box = locate_dofs_geometrical(V, lambda x: np.logical_and(np.logical_and(x[0] >= box[0][0],x[0] <= box[1][0]),
                                                                        np.logical_and(x[1] >= box[0][1],x[1] <= box[1][1])))
            u_unsafe = Function(V)
            u_unsafe.interpolate(lambda x: c_unsafe + 0*x[1])

            bc_unsafe = dirichletbc(u_unsafe, dofs_box)
            bcs.append(bc_unsafe)

        for box in target_box:
            dofs_box = locate_dofs_geometrical(V, lambda x: np.logical_and(np.logical_and(x[0] >= box[0][0],x[0] <= box[1][0]),
                                                                        np.logical_and(x[1] >= box[0][1],x[1] <= box[1][1])))
            
            u_safe = Function(V)
            u_safe.interpolate(lambda x: c_goal + 0*x[1])

            bc_safe = dirichletbc(u_safe, dofs_box)
            bcs.append(bc_safe)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        f = Constant(self.domain, ScalarType(f_val))

        a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = f * v * ufl.dx

        problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.uh = problem.solve()
        print("Solved the variational problem")

    def evaluate_solution(self,x,y):
        point = np.array([x,y])
        cell_candidates = geometry.compute_collisions(self.bb_tree, point)
        colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, point)
        return self.uh.eval(np.array([x,y,0]), colliding_cells[0])

    def evaluate_gradient(self,x,y):
        delta = 1
        point = np.array([x,y])
        cell_candidates = geometry.compute_collisions(self.bb_tree, point)
        colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, point)
        if len(colliding_cells) == 0:
            return 0,0
        grad_x = (self.uh.eval(np.array([x+delta,y,0]), colliding_cells[0]) - self.uh.eval(np.array([x,y,0]), colliding_cells[0])) # / delta
        grad_y = (self.uh.eval(np.array([x,y+delta,0]), colliding_cells[0]) - self.uh.eval(np.array([x,y,0]), colliding_cells[0])) # / delta
        return grad_x[0],grad_y[0]

    def plot_solution_with_trajectory(self,traj=[],show_edges=False,fig_name=None):
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'olive', 'brown', 'magenta', 'cyan', 'crimson','gray', 'black']

        u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(self.V)
        u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
        u_grid.point_data[""] = self.uh.x.array.real
        u_grid.set_active_scalars("")
        u_plotter = pyvista.Plotter()
        u_plotter.add_mesh(u_grid, show_edges=show_edges, scalar_bar_args={"position_x":0.2,"bold": True})

        for i in range(min(len(traj),len(colors))):
            trajectory = traj[i]
            trajectory[:,2] = 0
            u_plotter.add_points(trajectory, color=colors[i], point_size=5)
        # u_plotter.add_points(traj, color="blue", point_size=5, label='z(0)=0.35')
        # u_plotter.add_legend(size=(0.13,0.1),loc="lower right", face='-')

        # Show bounds
        u_plotter.show_bounds(xtitle='', ytitle='', ztitle='')

        u_plotter.view_xy()
        if not pyvista.OFF_SCREEN:
            u_plotter.show()


        # with u_plotter.window_size_context((768, 768)):
        if fig_name is not None:
            with u_plotter.window_size_context((768, 768)):
                u_plotter.save_graphic(fig_name)
