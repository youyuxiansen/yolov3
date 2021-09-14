from collections import Counter
import os
import argparse
import xml.etree.ElementTree as ET


def get_name_from_xml(xml_file: str) -> list:
	tree = ET.parse(xml_file)
	root = tree.getroot()
	cls_list = []
	cls_with_imagenames = {}
	for obj in root.iter('object'):
		cls = obj.find('name').text
		cls_list.append(cls)
	return cls_list


def replace_cls_name_xml(replace_relation: dict, xml_file: str) -> None:
	tree = ET.parse(xml_file)
	root = tree.getroot()
	for obj in root.iter('object'):
		try:
			print(obj.find('name').text, '\n', xml_file)
			obj.find('name').text = obj.find('name').text.replace(
				obj.find('name').text, replace_relation[obj.find('name').text])
		except (AttributeError, KeyError):
			pass
		tree.write(xml_file, encoding='utf-8')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--annatation-dir', '-a', type=str, help='Directory containing annatation files')
	parser.add_argument('--image-dir', '-i', type=str, help='Directory containing images')
	parser.add_argument('--print-cls', '-pc', action='store_true', help='Print unique class names')
	parser.add_argument('--replace-clsname', '-r', action='store_true',
	                    help='Specify config file for replacing class name in annatation')
	parser.add_argument('--print-class-distribution', '-pd', action='store_true',
	                    help='Print class distribution in all xmls')
	# TODO: needs support coco json files
	# parser.add_argument('--annatation_type', type=str, help='xml means VOC;json means coco')
	opt = parser.parse_args()
	xml_files = os.listdir(opt.annatation_dir)

	if opt.print_cls:
		allcls_list = []
		allcls_with_imagenames = {}
		for xml in xml_files:
			cls_list = get_name_from_xml(os.path.join(opt.annatation_dir, xml))
			allcls_list += cls_list

		print(set(allcls_list), '\n')
		if opt.print_class_distribution:
			print(Counter(allcls_list), '\n')

	if opt.replace_clsname:
		from config.cls_name_replacement_config import replace_relation
		for xml in xml_files:
			replace_cls_name_xml(replace_relation, os.path.join(opt.annatation_dir, xml))


