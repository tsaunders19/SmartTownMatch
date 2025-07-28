import React from 'react';
import Select, { components } from 'react-select';
import { FaHome, FaCity, FaTree } from 'react-icons/fa';

const options = [
  {
    value: 'Suburb',
    label: 'Suburb',
    icon: FaHome,
  },
  {
    value: 'City',
    label: 'City',
    icon: FaCity,
  },
  {
    value: 'Rural',
    label: 'Rural',
    icon: FaTree,
  },
];

const Option = (props) => {
  const { icon: Icon } = props.data;
  return (
    <components.Option {...props}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Icon color="#6c757d" />
        <span>{props.label}</span>
      </div>
    </components.Option>
  );
};

const SingleValue = (props) => {
  const { icon: Icon } = props.data;
  return (
    <components.SingleValue {...props}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Icon color="#6c757d" />
        <span>{props.data.label}</span>
      </div>
    </components.SingleValue>
  );
};

const customStyles = {
  control: (provided) => ({
    ...provided,
    backgroundColor: 'var(--form-background-color)',
    borderColor: 'var(--border-color)',
    paddingLeft: '4px',
    minHeight: '48px',
    color: 'var(--text-color)'
  }),
  option: (provided, state) => ({
    ...provided,
    backgroundColor: state.isSelected ? 'var(--primary-color)' : state.isFocused ? 'var(--hover-color)' : 'var(--form-background-color)',
    color: state.isSelected ? '#fff' : 'var(--text-color)',
  }),
  singleValue: (provided) => ({
    ...provided,
    color: 'var(--text-color)'
  }),
  placeholder: (provided) => ({
    ...provided,
    color: 'var(--text-color-secondary)'
  }),
  menu: (provided) => ({
    ...provided,
    animation: 'fadeSlide 150ms ease-out'
  }),
};

const LifestyleSelect = ({ value, onChange }) => {
  const selectedOption = options.find((o) => o.value === value);
  return (
    <Select
      options={options}
      value={selectedOption}
      onChange={(opt) => onChange(opt.value)}
      components={{ Option, SingleValue }}
      styles={customStyles}
      classNamePrefix="react-select"
      isSearchable={false}
    />
  );
};

export default LifestyleSelect; 