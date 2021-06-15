  pro make_cut_image
  image_address = 'C:\Users\User\Documents\sextractor-2.25.0\config'
  
  number_of_gals = 14721
  num_in_center = 0
  for p = 2, number_of_gals do begin
    object_num = 999
  ;read text document
  ;
  cd, image_address
  READCOL,STRCOMPRESS(STRING(p)+'-cat.txt', /REMOVE_ALL),F='F', number, xwin_image,ywin_image
  
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
 image_1 = READFITS(STRCOMPRESS(STRING(p) +'-check_segmented.fits', /REMOVE_ALL))
 image_2 = READFITS(STRCOMPRESS(STRING(p) +'-check_filtered.fits', /REMOVE_ALL))

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
 if num_in_center EQ 100 then begin
  print, "num in center = ", num_in_center
 endif
 
 if num_in_center EQ 500 then begin
   print, "num in center = ", num_in_center
 endif
 
 if num_in_center EQ 5000 then begin
   print, "num in center = ", num_in_center
 endif
 
 if num_in_center EQ 10000 then begin
   print, "num in center = ", num_in_center
 endif
 ;im3 = image(image_3)
 ;im2 = image(image_2)
 ;im1 = image(image_1)
 ;save image 3
 WRITEFITS, STRCOMPRESS('ordered_g_'+STRING(num_in_center) +'-check_filtered_cleaned.fits', /REMOVE_ALL),image_3
endfor
  end