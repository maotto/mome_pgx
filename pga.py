import chex
import flax
import hydra
import jax.numpy as jnp
import jax
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle
import time
import visu_brax

from dataclasses import dataclass
from functools import partial
from hydra.core.config_store import ConfigStore
from typing import Callable, Dict, Optional, Tuple, Any
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter
from qdax.core.emitters.mutation_operators import (
    polynomial_mutation, 
    polynomial_crossover, 
)
from qdax.core.emitters.pga_me_emitter import PGAMEEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.mome import MOME, MOMERepertoire
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.types import Fitness, Descriptor, RNGKey, ExtraScores
from qdax.utils.plotting import ( 
    plot_2d_map_elites_repertoire, 
    plot_map_elites_results,
)
from qdax.utils.metrics import CSVLogger, default_moqd_metrics


class RunPGA:

    """
    Args:

    """

    def __init__(self,   # Env config
                num_iterations: int, 
                num_init_cvt_samples: int,
                num_centroids: int,
                num_descriptor_dimensions: int,
                minval: int,
                maxval: int,
                batch_size: int, 
                scoring_fn: Callable,
                pg_emitter: PGAMEEmitter,
                episode_length: int,
                env_batch_size: int,
                metrics_fn: Callable,
                metrics_log_period: int,
                plot_repertoire_period: int,
                checkpoint_period: int,
                save_checkpoint_visualisations: bool,
                save_final_visualisations: bool,
                num_save_visualisations: int,

    ):
        self.num_iterations =  num_iterations
        self.num_init_cvt_samples = num_init_cvt_samples 
        self.num_centroids = num_centroids 
        self.num_descriptor_dimensions = num_descriptor_dimensions
        self.minval = minval
        self.maxval = maxval
        self.batch_size =  batch_size
        self.scoring_fn = scoring_fn
        self.pg_emitter = pg_emitter
        self.episode_length = episode_length
        self.env_batch_size = env_batch_size
        self.metrics_fn = metrics_fn
        self.metrics_log_period = metrics_log_period
        self.plot_repertoire_period = plot_repertoire_period
        self.checkpoint_period = checkpoint_period
        self.save_checkpoint_visualisations = save_checkpoint_visualisations
        self.save_final_visualisations = save_final_visualisations
        self.num_save_visualisations = num_save_visualisations


    def run(self,
            random_key: RNGKey,
            init_population: Any,
            env: Optional[Any]=None,
            policy_network: Optional[MLP]=None,
            ) -> Tuple[MOMERepertoire, jnp.ndarray, RNGKey]:

        # Set up logging functions 
        num_loops = self.num_iterations // self.metrics_log_period
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().handlers[0].setLevel(logging.INFO)
        logger = logging.getLogger(f"{__name__}")
        output_dir = "./" 

        # Name save directories
        repertoire_plots_save_dir = os.path.join(output_dir, "checkpoints", "repertoires", "plots")
        metrics_dir = os.path.join(output_dir, "checkpoints")
        final_metrics_dir = os.path.join(output_dir, "final", "metrics")
        final_plots_dir = os.path.join(output_dir, "final", "plots")
        final_repertoire_dir = os.path.join(output_dir, "final", "repertoire/")

        # Create save directories
        os.makedirs(repertoire_plots_save_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(final_metrics_dir, exist_ok=True)
        os.makedirs(final_plots_dir, exist_ok=True)
        os.makedirs(final_repertoire_dir, exist_ok=True)

        # Create visualisation directories
        if self.save_checkpoint_visualisations:
            visualisations_save_dir = os.path.join(output_dir, "checkpoints", "repertoires", "visualisations")
            os.makedirs(visualisations_save_dir)
            
        if self.save_final_visualisations:
            final_visualisation_dir = os.path.join(output_dir, "final", "visualisations")
            os.makedirs(final_visualisation_dir)

        # Instantiate MAP Elites
        map_elites = MAPElites(
            scoring_function=self.scoring_fn,
            emitter=self.pg_emitter,
            metrics_function=self.metrics_fn,
        )

        # Compute the centroids
        logger.warning("--- Computing the CVT centroids ---")

        # Start timing the algorithm
        init_time = time.time()
        centroids, random_key = compute_cvt_centroids(
            num_descriptors=self.num_descriptor_dimensions,
            num_init_cvt_samples=self.num_init_cvt_samples,
            num_centroids=self.num_centroids,
            minval=self.minval,
            maxval=self.maxval,
            random_key=random_key,
        )

        centroids_init_time = time.time() - init_time
        logger.warning(f"--- Duration for CVT centroids computation : {centroids_init_time:.2f}s ---")

        logger.warning("--- Algorithm initialisation ---")
        total_algorithm_duration = 0.0
        algorithm_start_time = time.time()

        # Compute initial repertoire
        repertoire, emitter_state, random_key = map_elites.init(
            init_population, centroids, random_key
        )

        initial_repertoire_time = time.time() - algorithm_start_time
        total_algorithm_duration += initial_repertoire_time
        logger.warning("--- Initialised initial repertoire ---")
        logger.warning("--- Starting the algorithm main process ---")

        timings = {"initial_repertoire_time": initial_repertoire_time,
                    "centroids_init_time": centroids_init_time,
                    "runtime_logs": jnp.zeros([(num_loops)+1, 1]),
                    "avg_iteration_time": 0.0,
                    "avg_evalps": 0.0}
       
        metrics_history = {
                "qd_score": jnp.array([0.0]), 
                "max_fitness": jnp.array([0.0]),  
                "coverage": jnp.array([0.0])}

        logger.warning(f"--- Running PGA for {num_loops} loops ---")
        
        # Run the algorithm
        for i in range(num_loops):
            iteration = (i+1) * self.metrics_log_period
            logger.warning(f"------ Iteration {iteration} out of {self.num_iterations} ------")
            start_time = time.time()

            # Log period number of QD itertions
            (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
                map_elites.scan_update,
                (repertoire, emitter_state, random_key),
                (),
                length=self.metrics_log_period,
            )

            timelapse = time.time() - start_time
            total_algorithm_duration += timelapse

            metrics_history = {key: jnp.concatenate((metrics_history[key], metrics[key])) for key in metrics}

            logger.warning(f"--- QD Score: {metrics['qd_score'][-1]:.2f}")
            logger.warning(f"--- Coverage: {metrics['coverage'][-1]:.2f}%")
            logger.warning(f"--- Max Fitness: {metrics['max_fitness'][-1]:.4f}")

            timings["avg_iteration_time"] = (timings["avg_iteration_time"]*(i*self.metrics_log_period) + timelapse) / ((i+1)*self.metrics_log_period)
            timings["avg_evalps"] = (timings["avg_evalps"]*(i*self.metrics_log_period) + ((self.batch_size*self.metrics_log_period)/timelapse)) / ((i+1)*self.metrics_log_period)
            timings["runtime_logs"] = timings["runtime_logs"].at[i, 0].set(total_algorithm_duration)

            # Save plot of repertoire every plot_repertoire_period iterations
            if iteration % self.plot_repertoire_period == 0:
                self.plot_repertoire(
                    repertoire,
                    centroids,
                    save_dir=repertoire_plots_save_dir,
                    save_name=f"{iteration}",
                )
    
            # Save latest repertoire and metrics every 'checkpoint_period' iterations
            if iteration % self.checkpoint_period == 0:
                repertoire.save(path=final_repertoire_dir)
                    
                with open(os.path.join(metrics_dir, "metrics_history.pkl"), 'wb') as f:
                    pickle.dump(metrics_history, f)

                with open(os.path.join(metrics_dir, "timings.pkl"), 'wb') as f:
                    pickle.dump(timings, f)
                
                if self.save_checkpoint_visualisations:
                    random_key, subkey = jax.random.split(random_key)
                    visu_brax.save_samples(
                        env,
                        policy_network,
                        subkey,
                        repertoire, 
                        self.num_save_visualisations,
                        iteration,
                        save_dir=visualisations_save_dir,
                    )
        

        print("REPERTOIRE TYPE:", type(repertoire))
        total_duration = time.time() - init_time

        logger.warning("--- FINAL METRICS ---")
        logger.warning(f"Total duration: {total_duration:.2f}s")
        logger.warning(f"Main algorithm duration: {total_algorithm_duration:.2f}s")
        logger.warning(f"Max fitness: {metrics['max_fitness'][-1]:.2f}%")
        logger.warning(f"QD Score: {metrics['qd_score'][-1]:.2f}")
        logger.warning(f"Coverage: {metrics['coverage'][-1]:.2f}%")

        # Save metrics
        with open(os.path.join(metrics_dir, "metrics_history.pkl"), 'wb') as f:
            pickle.dump(metrics_history, f)

        with open(os.path.join(metrics_dir, "timings.pkl"), 'wb') as f:
            pickle.dump(timings, f)

        with open(os.path.join(final_metrics_dir, "final_metrics.pkl"), 'wb') as f:
            pickle.dump(final_metrics_dir, f)
        
        
        # Save final repertoire
        repertoire.save(path=final_repertoire_dir)

        # Save visualisation of best repertoire
        if self.save_final_visualisations:
            random_key, subkey = jax.random.split(random_key)
            
            visu_brax.save_samples(
                env,                       
                policy_network,
                subkey,
                repertoire, 
                self.num_save_visualisations,
                save_dir=final_visualisation_dir,
            )

        # Save final plots 
        self.plot_scores_evolution(
            repertoire,
            metrics_history,
            save_dir=final_plots_dir
        )

        self.plot_repertoire(
            repertoire,
            centroids,
            save_dir=final_plots_dir,
            save_name="final",
        )

    
    def plot_scores_evolution(
        self,
        repertoire: Any,
        metrics_history: Dict,
        save_dir: str="./",
    ) -> None:
        
        # +1 to iterations as we set metrics = 0 for iteration 0
        env_steps = jnp.arange(self.num_iterations + 1) * self.episode_length * self.env_batch_size

        fig, axes = plot_map_elites_results(env_steps=env_steps,
            metrics=metrics_history, 
            repertoire=repertoire, 
            min_bd=self.minval, 
            max_bd=self.maxval,
        )

        plt.savefig(os.path.join(save_dir, "pga_scores_evolution.png"))
        plt.close()

    def plot_repertoire(
        self,
        repertoire: MapElitesRepertoire,
        centroids: jnp.ndarray,
        save_dir: str="./",
        save_name: str="",
    ) -> None:
        
        fig, axes = plt.subplots(figsize=(18, 6))

        # add map elites plot on last axes
        fig, axes = plot_2d_map_elites_repertoire(
            centroids=centroids,
            repertoire_fitnesses=repertoire.fitnesses,
            minval=self.minval,
            maxval=self.maxval,
            repertoire_descriptors=repertoire.descriptors,
            ax=axes,
        )

        plt.savefig(os.path.join(save_dir, f"repertoire_{save_name}"))
        plt.close()

    