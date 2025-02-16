# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: uploadPic.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

from grpc_tools import protoc


DESCRIPTOR = _descriptor.FileDescriptor(
  name='uploadPic.proto',
  package='namespaceUploadpic',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0fuploadPic.proto\x12\x12namespaceUploadpic\"z\n\x08MatImage\x12\x0c\n\x04rows\x18\x01 \x01(\x05\x12\x0c\n\x04\x63ols\x18\x02 \x01(\x05\x12\x10\n\x08\x65lt_type\x18\x03 \x01(\x05\x12\x10\n\x08\x63hannels\x18\x04 \x01(\x05\x12\x10\n\x08mat_data\x18\x05 \x01(\x0c\x12\n\n\x02id\x18\x06 \x01(\x05\x12\x10\n\x08video_id\x18\x07 \x01(\x05\"=\n\x05Reply\x12\x0c\n\x04\x42\x62ox\x18\x01 \x03(\x05\x12\x0f\n\x07message\x18\x02 \x01(\t\x12\x15\n\rrequest_state\x18\x03 \x01(\x08\x32\x9e\x01\n\x11uploadPicServicer\x12\x43\n\x06Upload\x12\x1c.namespaceUploadpic.MatImage\x1a\x19.namespaceUploadpic.Reply\"\x00\x12\x44\n\x07GetBbox\x12\x1c.namespaceUploadpic.MatImage\x1a\x19.namespaceUploadpic.Reply\"\x00\x62\x06proto3'
)




_MATIMAGE = _descriptor.Descriptor(
  name='MatImage',
  full_name='namespaceUploadpic.MatImage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='rows', full_name='namespaceUploadpic.MatImage.rows', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cols', full_name='namespaceUploadpic.MatImage.cols', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='elt_type', full_name='namespaceUploadpic.MatImage.elt_type', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='channels', full_name='namespaceUploadpic.MatImage.channels', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mat_data', full_name='namespaceUploadpic.MatImage.mat_data', index=4,
      number=5, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='id', full_name='namespaceUploadpic.MatImage.id', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='video_id', full_name='namespaceUploadpic.MatImage.video_id', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=39,
  serialized_end=161,
)


_REPLY = _descriptor.Descriptor(
  name='Reply',
  full_name='namespaceUploadpic.Reply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='Bbox', full_name='namespaceUploadpic.Reply.Bbox', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='message', full_name='namespaceUploadpic.Reply.message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='request_state', full_name='namespaceUploadpic.Reply.request_state', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=163,
  serialized_end=224,
)

DESCRIPTOR.message_types_by_name['MatImage'] = _MATIMAGE
DESCRIPTOR.message_types_by_name['Reply'] = _REPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

MatImage = _reflection.GeneratedProtocolMessageType('MatImage', (_message.Message,), {
  'DESCRIPTOR' : _MATIMAGE,
  '__module__' : 'uploadPic_pb2'
  # @@protoc_insertion_point(class_scope:namespaceUploadpic.MatImage)
  })
_sym_db.RegisterMessage(MatImage)

Reply = _reflection.GeneratedProtocolMessageType('Reply', (_message.Message,), {
  'DESCRIPTOR' : _REPLY,
  '__module__' : 'uploadPic_pb2'
  # @@protoc_insertion_point(class_scope:namespaceUploadpic.Reply)
  })
_sym_db.RegisterMessage(Reply)



_UPLOADPICSERVICER = _descriptor.ServiceDescriptor(
  name='uploadPicServicer',
  full_name='namespaceUploadpic.uploadPicServicer',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=227,
  serialized_end=385,
  methods=[
  _descriptor.MethodDescriptor(
    name='Upload',
    full_name='namespaceUploadpic.uploadPicServicer.Upload',
    index=0,
    containing_service=None,
    input_type=_MATIMAGE,
    output_type=_REPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetBbox',
    full_name='namespaceUploadpic.uploadPicServicer.GetBbox',
    index=1,
    containing_service=None,
    input_type=_MATIMAGE,
    output_type=_REPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_UPLOADPICSERVICER)

DESCRIPTOR.services_by_name['uploadPicServicer'] = _UPLOADPICSERVICER

# @@protoc_insertion_point(module_scope)
