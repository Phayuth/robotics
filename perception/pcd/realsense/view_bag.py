import open3d as o3d

bag_filename = '/home/yuth/ws_yuthdev/robotics_manipulator/map_pcd/test_bag.bag'

bag_reader = o3d.t.io.RSBagReader()
bag_reader.open(bag_filename)
im_rgbd = bag_reader.next_frame()
print(im_rgbd)

while not bag_reader.is_eof():
    # process im_rgbd.depth and im_rgbd.color
    im_rgbd = bag_reader.next_frame()

bag_reader.close()