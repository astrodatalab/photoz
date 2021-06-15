pro make_cut_image
  image_address = 'C:\Users\User\Documents\sextractor-2.25.0\config\y_stuff_full'

  number_of_gals = 52592
  num_in_center = 0
  for p = 2, number_of_gals do begin
    object_num = 999
    ;read text document
    ;
    cd, image_address
    READCOL,STRCOMPRESS(STRING(p)+'-y_cat.txt', /REMOVE_ALL),F='F', number, xwin_image,ywin_image

    n_rows = n_elements(number)
    for i = 0, n_rows-1 do begin
      if xwin_image[i] GE 55 AND xwin_image[i] LE 65 then begin
        if ywin_image[i] GE 55 AND ywin_image[i] LE 65 then begin
          num_in_center++
          object_num = i+1
        endif
      endif
    endfor

    ;print,object_num
    if object_num EQ 999 then continue


    ;read image_1:
    image_1 = READFITS(STRCOMPRESS(STRING(p) +'-y_check_segmented.fits', /REMOVE_ALL))
    image_2 = READFITS(STRCOMPRESS(STRING(p) +'-y_check_filtered.fits', /REMOVE_ALL))
    if n_elements(image_1) NE 14400 then continue
    if n_elements(image_2) NE 14400 then continue
    ;image_1 = READFITS('4-check.fits')
    ; image_2 = READFITS('4-check_filtered.fits')
    image_3 = make_array(120,120, /float)
    ;create new segmented image with only 1 object:
    for j = 0,120-1 do begin
      for i = 0, 120-1 do begin
        pixel_i_j = image_1[i,j]
        if pixel_i_j EQ object_num then begin
          image_3[i,j] = image_2[i,j]
        endif
      endfor
    endfor

    ;endif
    ;im3 = image(image_3)
    ;im2 = image(image_2)
    ;im1 = image(image_1)
    ;save image 3
    WRITEFITS, STRCOMPRESS('unordered_y_'+STRING(num_in_center) +'-check_filtered_cleaned.fits', /REMOVE_ALL),image_3
  endfor
  
  print, num_in_center
end