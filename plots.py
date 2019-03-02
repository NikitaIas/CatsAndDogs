import matplotlib.pyplot as plot
import matplotlib.patches as patches

def show_img(image_array, label_array, size):
    #show one image
    plot.subplot(1,1,1)
    plot.imshow(image_array)

    category,x1,y1,x2,y2 = label_array.flatten()
    x1 = size[0] * x1
    x2 = size[0] * x2
    y1 = size[1] * y1
    y2 = size[1] * y2

    rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='r',facecolor='none')

    ref = plot.gca()
    ref.text(5, 5, 'CAT'if category<0.5 else 'DOG',fontsize=12,
                horizontalalignment='left',color='r' if category<0.5 else 'b',
                verticalalignment='top')
    ref.add_patch(rect)
    plot.show()

def show_images(image_arrays, size, label_arrays, categories=True):
    # show 10 images
    plot.figure(figsize=(12, 9), dpi=80)
    for i in range(10):
        plot.subplot(2,5,i+1)
        plot.imshow(image_arrays[i])
        if categories:
            category,x1,y1,x2,y2 = label_arrays[i].flatten()
        else:
            x1,y1,x2,y2 = label_arrays[i].flatten()
        x1 = size[0] * x1
        x2 = size[0] * x2
        y1 = size[1] * y1
        y2 = size[1] * y2
        if categories:
            rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,
                    edgecolor='r' if category>0.5 else 'b',facecolor='none')
        else:
            rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,
                    edgecolor='b',facecolor='none')
        ref = plot.gca()
        if categories:
            ref.text(5, 5, 'CAT'if category<0.5 else 'DOG',fontsize=12,
                horizontalalignment='left',color='r' if category<0.5 else 'b',
                verticalalignment='top')
        ref.add_patch(rect)
    plot.show()

def show_images_class(image_arrays, label_arrays):
    # show images and categories
    plot.figure(figsize=(12, 9), dpi=80)
    for i in range(10):
        plot.subplot(2,5,i+1)
        plot.imshow(image_arrays[i])
        category = label_arrays[i]
        ref = plot.gca()
        ref.text(5, 5, 'CAT'if category<0.5 else 'DOG',fontsize=12,
            horizontalalignment='left',color='r' if category<0.5 else 'b',
            verticalalignment='top')
    plot.show()

def show_bounds(imgs,size,lbls_pred,lbls_orig):
    # show 10 images with 2 sets of boxes
    plot.figure(figsize=(12, 9), dpi=80)
    for i in range(10):
        plot.subplot(2,5,i+1)
        plot.imshow(imgs[i])
        x1,y1,x2,y2 = lbls_orig[i].flatten()
        p_x1, p_y1, p_x2, p_y2 = lbls_pred[i].flatten()
        x1 = size[0] * x1
        x2 = size[0] * x2
        y1 = size[1] * y1
        y2 = size[1] * y2
        p_x1 = size[0] * p_x1
        p_x2 = size[0] * p_x2
        p_y1 = size[1] * p_y1
        p_y2 = size[1] * p_y2
        rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,
                    edgecolor='y',facecolor='none')
        rect_pred = patches.Rectangle((p_x1,p_y1),p_x2-p_x1,p_y2-p_y1,linewidth=1,
                    edgecolor='r',facecolor='none')
        ref = plot.gca()
        ref.add_patch(rect)
        ref.add_patch(rect_pred)
    plot.show()

def show_all(imgs,size,lbls_pred,lbls_orig):
    # show 10 images with 2 sets of boxes and 2 category labels
    plot.figure(figsize=(12, 9), dpi=80)
    for i in range(10):
        plot.subplot(2,5,i+1)
        plot.imshow(imgs[i])
        cat, x1,y1,x2,y2 = lbls_orig[i].flatten()
        p_cat, p_x1, p_y1, p_x2, p_y2 = lbls_pred[i].flatten()
        x1 = size[0] * x1
        x2 = size[0] * x2
        y1 = size[1] * y1
        y2 = size[1] * y2
        p_x1 = size[0] * p_x1
        p_x2 = size[0] * p_x2
        p_y1 = size[1] * p_y1
        p_y2 = size[1] * p_y2
        rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,
                    edgecolor='y',facecolor='none')
        rect_pred = patches.Rectangle((p_x1,p_y1),p_x2-p_x1,p_y2-p_y1,linewidth=1,
                    edgecolor='r',facecolor='none')

        ref = plot.gca()
        ref.text(5, 5, 'CAT'if cat<0.5 else 'DOG',fontsize=12,
            horizontalalignment='left',color='r' if cat<0.5 else 'b',
            verticalalignment='top')
        ref.text(5, 5, 'CAT'if p_cat<0.5 else 'DOG',fontsize=12,
            horizontalalignment='right',color='r' if p_cat<0.5 else 'b',
            verticalalignment='top')
        ref.add_patch(rect)
        ref.add_patch(rect_pred)
    plot.show()
    

def show_unlabeled(imgs,size,lbls_pred):
    # show 10 images with 2 sets of boxes and 2 category labels
    plot.figure(figsize=(12, 9), dpi=80)
    for i in range(10):
        plot.subplot(2,5,i+1)
        plot.imshow(imgs[i])
        p_cat, p_x1, p_y1, p_x2, p_y2 = lbls_pred[i].flatten()
        p_x1 = size[0] * p_x1
        p_x2 = size[0] * p_x2
        p_y1 = size[1] * p_y1
        p_y2 = size[1] * p_y2
        rect_pred = patches.Rectangle((p_x1,p_y1),p_x2-p_x1,p_y2-p_y1,linewidth=1,
                    edgecolor='r',facecolor='none')

        ref = plot.gca()
        ref.text(5, 5, 'CAT'if p_cat<0.5 else 'DOG',fontsize=12,
            horizontalalignment='right',color='r' if p_cat<0.5 else 'b',
            verticalalignment='top')
        ref.add_patch(rect_pred)
    plot.show()
    