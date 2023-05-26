function sysCall_info()
    return {autoStart=false}
end

function sysCall_init()
    -- start of simulation
	print("starting data collection")

	sim.loadScene('/home/yuth/mycustomscence.ttt')
	data = {} -- save data
	seq = {0.1,0.2,0.3} -- data to loop through
    i = 1 -- sequence increment
end

function sysCall_addOnScriptSuspend()
    print("Ending... (triggered by the user)")
    return {cmd='cleanup'} -- end this add-on. The cleanup section will be called
end

function sysCall_cleanup()
    print("End All sequence")

    -- save data to file
    file = io.open("my_data.csv", "w")
    for i = 1, #data do
        file:write(table.concat(data[i], ",") .. "\n")
    end
    file:close()

    print("Executing the clean-up section")
end

function sysCall_beforeSimulation()
    -- is executed before a simulation starts
    print("sequence number = ",i)

    cub = sim.getObject('./Cuboid')
    x = seq[i]
    y = seq[i]
    sim.setObjectPosition(cub,-1,{x,y,2})

    --print(sim.getObjectPosition(cub,-1))
end

function sysCall_afterSimulation()
    -- is executed before a simulation ends
    -- increment sequence by 1
    i = i+1
end

function sysCall_actuation()
    -- during simulation
    t = sim.getSimulationTime()

    if t >= 5 then
        pose = sim.getObjectPosition(cub,-1)
        print(pose)

        data[#data+1] = pose -- save data
        
        sim.stopSimulation()
    end
end

function sysCall_nonSimulation()
    -- when the simulation is not running, start the simulation
	sim.startSimulation()
end
