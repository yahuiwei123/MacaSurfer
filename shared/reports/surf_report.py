from pathlib import Path
from nibabel import load
from surfplot import Plot
import numpy as np
from niworkflows.viz.utils import (
    compose_view,
    cuts_from_bbox,
    plot_registration,
)

class SurfacePlotter:
    """A class for creating various surface-based visualizations.
    
    This class handles:
    1. Surface contour visualization on volumetric images
    2. Parcellation visualization on surfaces
    3. Surface metric visualization (curvature, sulci, thickness)
    """
    
    def __init__(self, subjects_dir, subject_id, out_report, compress_report=True):
        """Initialize the SurfacePlotter.
        
        Parameters
        ----------
        subjects_dir : str
            Path to FreeSurfer subjects directory
        subject_id : str
            Subject identifier
        out_report : str
            Output path for the final report
        compress_report : bool, optional
            Whether to compress the report, by default True
        """
        self.subjects_dir = Path(subjects_dir)
        self.subject_id = subject_id
        self.out_report = out_report
        self.compress_report = compress_report
        self.rootdir = self.subjects_dir / self.subject_id
        
    def plot_surface_contour_volume(self, runtime=None):
        """Plot surface contours on volumetric images.
        
        Parameters
        ----------
        runtime : object, optional
            NiPype runtime object, by default None
            
        Returns
        -------
        dict
            Dictionary containing the output report path
        """
        _anat_file = str(self.rootdir / 'Results' / 'ACPC' / 'mri' / 'T1w_acpc_brain.nii.gz')
        _contour_file = str(self.rootdir / 'Results' / 'ACPC' / 'mri' / 'ribbon.nii.gz')

        anat = load(_anat_file)
        contour_nii = load(_contour_file)

        n_cuts = 7
        cuts = cuts_from_bbox(contour_nii, cuts=n_cuts)

        # Call composer
        out_files = plot_registration(
            anat,
            'fixed-image',
            estimate_brightness=True,
            cuts=cuts,
            contour=contour_nii,
            compress=self.compress_report,
        )
        return out_files

    def plot_metrics_on_surface(self, surface_type='very_inflated', hemisphere='L', 
                               metric='curvature', view='lateral'):
        """Plot surface metrics (curvature, sulci, thickness) on surfaces.
        
        Parameters
        ----------
        surface_type : str, optional
            Surface type (e.g., 'pial', 'white'), by default 'pial'
        hemisphere : str, optional
            Hemisphere ('left' or 'right'), by default 'left'
        metric : str, optional
            Metric to plot ('curvature', 'sulci', 'thickness'), by default 'curvature'
        view : str, optional
            View ('lateral', 'medial'), by default 'lateral'
            
        Returns
        -------
        surfplot.Plot
            The created surface plot
        """
        # Validate inputs
        if surface_type not in ['white', 'pial', 'inflated', 'very_inflated', 'sphere']:
            raise ValueError("surface_type must be 'white', 'pial', 'inflated', 'very_inflated' or 'sphere'")
        if hemisphere not in ['L', 'R']:
            raise ValueError("hemisphere must be 'L' or 'R'")
        if metric not in ['curvature', 'sulc', 'thickness']:
            raise ValueError("metric must be 'curvature', 'sulc', or 'thickness'")
            
        # Load surface data
        surf_file = str(self.rootdir / 'Results' / 'ACPC' / 'Native' / f'{hemisphere}.{surface_type}.surf.gii')
        metric_file = str(self.rootdir / 'Results' / 'ACPC' / 'Native' / f'{hemisphere}.{metric}.shape.gii')
        
        # Create plot
        p = Plot(surf_file)
        
        # Add the metric data
        p.add_layer(metric_file, cmap='gray', color_range=(-1, 1))
        
        return p

    def plot_parcellation_on_surface(self, surface_type='pial', 
                                    hemisphere='L', view='lateral'):
        """Plot parcellation on surface.
        
        Parameters
        ----------
        surface_type : str, optional
            Surface type (e.g., 'pial', 'white'), by default 'pial'
        hemisphere : str, optional
            Hemisphere ('left' or 'right'), by default 'left'
        view : str, optional
            View ('lateral', 'medial'), by default 'lateral'
            
        Returns
        -------
        surfplot.Plot
            The created surface plot
        """
        # Validate inputs
        if surface_type not in ['white', 'pial', 'inflated', 'very_inflated', 'sphere']:
            raise ValueError("surface_type must be 'white', 'pial', 'inflated', 'very_inflated' or 'sphere'")
        if hemisphere not in ['L', 'R']:
            raise ValueError("hemisphere must be 'L' or 'R'")
            
        # Load surface and parcellation data
        surf_file = str(self.rootdir / 'Results' / 'ACPC' / 'Native' / f'{hemisphere}.{surface_type}.surf.gii')
        parc_data = load(str(self.rootdir / 'Results' / 'ACPC' / 'Native' / f'{hemisphere}.aparc.label.gii')).get_fdata()
        
        # Create plot
        p = Plot(surf_file)
        
        # Add parcellation (assuming parc_data is a label map)
        unique_labels = np.unique(parc_data)
        p.add_layer(parc_data, cmap='tab20', color_range=(0, len(unique_labels)))
            
        return p

    def plot_all_then_save(self, output_file=None, **kwargs):
        """Compose all plots and save to a single file.
        
        Parameters
        ----------
        output_file : str, optional
            Output file path, by default uses the initialized out_report
        **kwargs
            Additional arguments to pass to individual plot functions
        """
        output_file = output_file or self.out_report
        
        # Example of adding other plots (customize as needed)
        metrics = ['curvature', 'sulc', 'thickness']
        hemispheres = ['L', 'R']
        
        combined_plot = compose_view(
            self.plot_surface_contour_volume(),
            
            [self.plot_surface_contour_volume(hemisphere=h, metric=m, view='lateral') 
             for m in metrics 
             for h in hemispheres],
            
            [self.plot_parcellation_on_surface(hemisphere=h, view='lateral') 
             for h in hemispheres],
            
            out_file=output_file
        )
        
        return combined_plot