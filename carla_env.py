import random
import carla
import config

class CarlaEnvironment:
    def __init__(self):
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.camera = None
        self.lidar = None
        self.actor_list = []
        self.original_settings = None

        self.connect()

    def connect(self):
        print(f"Connecting to CARLA server at {config.HOST}:{config.PORT}...")
        self.client = carla.Client(config.HOST, config.PORT)
        self.client.set_timeout(config.TIMEOUT)
        
        # Load Map
        self.world = self.client.get_world()
        if self.world.get_map().name.split('/')[-1] != config.TOWN:
            print(f"Loading {config.TOWN}...")
            self.client.load_world(config.TOWN)
            self.world = self.client.get_world()
        
        print(f"Connected to {self.world.get_map().name}")
        
        # Setup synchronous mode
        if config.SYNC_MODE:
            self.original_settings = self.world.get_settings()
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = config.FIXED_DELTA_SECONDS
            self.world.apply_settings(settings)
            print(f"Synchronous mode enabled ({1.0/config.FIXED_DELTA_SECONDS} Hz)")

    def spawn_ego_vehicle(self):
        blueprint_library = self.world.get_blueprint_library()
        ego_bp = blueprint_library.find(config.VEHICLE_MODEL)
        ego_bp.set_attribute('role_name', 'ego')

        spawn_points = self.world.get_map().get_spawn_points()
        
        for spawn_point in spawn_points:
            self.ego_vehicle = self.world.try_spawn_actor(ego_bp, spawn_point)
            if self.ego_vehicle is not None:
                self.actor_list.append(self.ego_vehicle)
                print(f"Spawned ego vehicle at: {spawn_point.location}")
                break
        
        if self.ego_vehicle is None:
            raise RuntimeError("Could not find a valid spawn point for the ego vehicle.")

    def spawn_camera(self, callback):
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(config.CAMERA_RES_X))
        camera_bp.set_attribute('image_size_y', str(config.CAMERA_RES_Y))
        camera_bp.set_attribute('fov', str(config.CAMERA_FOV))
        
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(lambda data: callback(data))
        print("Spawned RGB Camera.")

    def spawn_lidar(self, callback):
        blueprint_library = self.world.get_blueprint_library()
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(config.LIDAR_RANGE))
        lidar_bp.set_attribute('rotation_frequency', str(config.LIDAR_ROTATION_FREQ))
        lidar_bp.set_attribute('channels', str(config.LIDAR_CHANNELS))
        lidar_bp.set_attribute('points_per_second', str(config.LIDAR_POINTS_PER_SEC))
        
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_vehicle)
        self.actor_list.append(self.lidar)
        self.lidar.listen(lambda data: callback(data))
        print("Spawned LiDAR.")

    def spawn_npc_vehicles(self, num_vehicles=None):
        """
        Spawns NPC vehicles with autopilot enabled via Traffic Manager.
        They will drive around the map and give YOLO targets to detect.
        """
        if num_vehicles is None:
            num_vehicles = config.NPC_VEHICLES_COUNT

        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        # Exclude bikes and motorcycles which are less stable with TM
        vehicle_blueprints = [bp for bp in vehicle_blueprints
                              if int(bp.get_attribute('number_of_wheels')) == 4]

        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        # Skip spawn point 0 — that's where ego vehicle likely is
        available_points = [sp for sp in spawn_points
                            if sp.location.distance(self.ego_vehicle.get_location()) > 10.0]

        spawned = 0
        for sp in available_points[:num_vehicles * 2]:  # try twice as many to handle occupied spots
            if spawned >= num_vehicles:
                break
            bp = random.choice(vehicle_blueprints)
            npc = self.world.try_spawn_actor(bp, sp)
            if npc is not None:
                npc.set_autopilot(True)   # Traffic Manager handles driving
                self.actor_list.append(npc)
                spawned += 1

        print(f"Spawned {spawned} NPC vehicles with autopilot.")

    def spawn_lead_vehicle(self, distance_ahead=20.0):
        """
        Spawns one vehicle directly ahead of the ego vehicle at a fixed distance.
        This vehicle drives slowly, giving YOLO a guaranteed detection target.
        """
        blueprint_library = self.world.get_blueprint_library()
        # Use a clearly visible vehicle
        lead_bp = blueprint_library.find('vehicle.tesla.model3')

        # Get ego transform and calculate a point ahead
        ego_tf = self.ego_vehicle.get_transform()
        fwd = ego_tf.get_forward_vector()
        spawn_loc = carla.Location(
            x=ego_tf.location.x + fwd.x * distance_ahead,
            y=ego_tf.location.y + fwd.y * distance_ahead,
            z=ego_tf.location.z + 0.5,
        )
        spawn_tf = carla.Transform(spawn_loc, ego_tf.rotation)

        lead = self.world.try_spawn_actor(lead_bp, spawn_tf)
        if lead is not None:
            # Keep it stationary (no autopilot) so it stays in camera FOV for testing
            self.actor_list.append(lead)
            print(f"Spawned stationary lead vehicle {distance_ahead}m ahead of ego.")
        else:
            print("Warning: Could not spawn lead vehicle at that location (occupied). "
                  "NPC traffic will still appear.")

    def tick(self):
        if config.SYNC_MODE:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def cleanup(self):
        print("\nCleaning up actors...")
        if self.original_settings:
            self.world.apply_settings(self.original_settings)
            
        for actor in reversed(self.actor_list):
            if actor.is_alive:
                actor.destroy()
        print("Cleanup complete.")
