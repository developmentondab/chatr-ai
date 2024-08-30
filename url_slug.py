import re

def get_string_from_url(url):

    # Regular expression pattern to extract the site name (hostname)
    site_name_pattern = r'https?://([^/]+)/'
    site_name_match = re.match(site_name_pattern, url)

    # Regular expression pattern to extract the last URL parameter name
    last_param_pattern = r'/([^/]+)/?$'
    last_param_match = re.search(last_param_pattern, url)

    # Extract the results
    site_name = site_name_match.group(1) if site_name_match else None
    last_parameter_name = last_param_match.group(1) if last_param_match else None

    if site_name and last_parameter_name and site_name !=last_parameter_name:
        sr = ".".join([site_name,last_parameter_name])
    else:
        sr = last_parameter_name
        
    return sr
    
def slugify(s):
  s = s.lower().strip()
  
  s = re.sub(r'[\s.]+', '-', s)  
  s = re.sub(r'[^\w\s-]', '', s) 
  s = re.sub(r'[\s_-]+', '-', s)  
  s = re.sub(r'^-+|-+$', '', s)
  return s

def get_slug_from_url(url):

    url_arr = []
    if isinstance(url, str):
        url_arr.append(url)
        url_arr = url
    
    converted_string = get_string_from_url(url[0])
    sluged_str = slugify(converted_string)
    return sluged_str, url_arr

